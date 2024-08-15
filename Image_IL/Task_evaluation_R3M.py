
import sys
path_to_add = '/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2'
sys.path.append(path_to_add)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image as RosImage
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
import time
from r3m import load_r3m
import argparse
import os
import importlib

parser = argparse.ArgumentParser(description='Behavior Cloning Testing')
parser.add_argument('--task_name', type=str, required=True, help='Name of the task')
parser.add_argument('--view_name', type=str, required=True, help='Name of the view')
parser.add_argument('--trans_error', type=float, required=True, help='Translational error threshold in cm')
parser.add_argument('--angle_error', type=float, required=True, help='Angular error threshold in degrees')
args = parser.parse_args()

task_name = args.task_name
view_name = args.view_name


module_name = f"{args.task_name.capitalize()}_env"
class_name = f"SRC_{args.task_name.lower()}"
module = importlib.import_module(module_name)
SRC_test = getattr(module, class_name)

if task_name == "Approach" or task_name == "Regrasp":
    is_grasp = False
else:
    is_grasp = True

print(f"is grasp: {is_grasp}")

r3m_model = load_r3m("resnet50")
r3m_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r3m_model.to(device)

class BehaviorCloningModel(nn.Module):
    def __init__(self, r3m):
        super(BehaviorCloningModel, self).__init__()
        self.r3m = r3m
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(2048 + 7),
            nn.Linear(2048 + 7, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
            nn.Tanh()
        ).to(device)

    def forward(self, x, proprioceptive_data):
        with torch.no_grad():
            visual_features = self.r3m(x)
        combined_input = torch.cat((visual_features, proprioceptive_data), dim=1)
        return self.regressor(combined_input)

def load_r3m_model(model_path, r3m):
    model = BehaviorCloningModel(r3m).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_action(model, image_np, proprio_data):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image_np).unsqueeze(0).to(device)
    proprioceptive_tensor = torch.tensor(proprio_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_action = model(image, proprioceptive_tensor)
    return predicted_action.cpu().numpy()

current_images = {}
image_received = {}
bridge = CvBridge()

def image_callback(msg, camera_id):
    global current_images, image_received
    try:
        current_images[camera_id] = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_received[camera_id] = True
    except Exception as e:
        rospy.logerr(f"Failed to convert image from {camera_id}: {e}")

camera_topics = {
    view_name: f'/ambf/env/cameras/cameraL/ImageData' if view_name == 'front' else '/ambf/env/cameras/normal_camera/ImageData'
}

for cam_id, topic in camera_topics.items():
    rospy.Subscriber(topic, RosImage, image_callback, callback_args=cam_id)
    image_received[cam_id] = False

def wait_for_images():
    rate = rospy.Rate(100)
    while not all(image_received.values()) and not rospy.is_shutdown():
        rate.sleep()
    for key in image_received:
        image_received[key] = False

model_path = f'/home/jwu220/Trajectory_cloud/IL_model_v2/{task_name}/R3M_{view_name}_view/Model/model_final.pth'
model = load_r3m_model(model_path, r3m_model)

seed = 60
set_random_seed(seed)
max_episode_steps = 200
trans_step = 5e-4
angle_step = np.deg2rad(2)
jaw_step = 0.05
threshold = [args.trans_error,np.deg2rad(args.angle_error)]
step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step], dtype=np.float32)

gym.envs.register(id="SAC_HER_sparse", entry_point=SRC_test, max_episode_steps=max_episode_steps)
env = gym.make("SAC_HER_sparse", render_mode=None, reward_type="dense", seed=seed, threshold=threshold, max_episode_step=max_episode_steps, step_size=step_size)

all_trajectory_lengths = []
all_time_costs = []
success_rates = []

def run_experiment(num_episodes=20):
    total_length = 0
    total_timecost = 0
    total_success = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        time.sleep(0.5)
        trajectory_length = 0
        for timestep in range(max_episode_steps):
            wait_for_images()
            proprio_data = obs["observation"][0:7]
            action = predict_action(model, current_images[view_name], proprio_data).squeeze()
            action[0:3] = action[0:3] + np.random.uniform(-0.1, 0.1, size=action[0:3].shape)
            if is_grasp:
                action[-1] = 0.0
            next_obs, reward, done, _, info = env.step(action)
            trajectory_length += np.linalg.norm(action[0:3] * trans_step * 1000)
            time.sleep(0.01)
            obs = next_obs
            if done:
                total_success += 1
                total_length += trajectory_length
                total_timecost += timestep + 1
                all_trajectory_lengths.append(trajectory_length)
                all_time_costs.append(timestep + 1)
                print(f"Episode {episode + 1} completed in {timestep + 1} steps")
                break
        if not done:
            print(f"Episode {episode + 1} did not complete successfully")
    
    success_rate = total_success / num_episodes
    avg_length = total_length / total_success if total_success > 0 else 0
    avg_timecost = total_timecost / total_success if total_success > 0 else 0
    return success_rate, avg_length, avg_timecost

num_experiments = 3
num_episodes = 20

for i in range(num_experiments):
    print(f"\nRunning experiment {i+1}/{num_experiments}")
    success_rate, avg_length, avg_timecost = run_experiment(num_episodes)
    success_rates.append(success_rate)
    print(f"Experiment {i+1} completed")

mean_success_rate = np.mean(success_rates)
std_success_rate = np.std(success_rates)

mean_avg_length = np.mean(all_trajectory_lengths)
std_avg_length = np.std(all_trajectory_lengths)
mean_avg_timecost = np.mean(all_time_costs)
std_avg_timecost = np.std(all_time_costs)

print("\nFinal Results:")
print(f"Success Rate: {mean_success_rate:.2%} ± {std_success_rate:.2%}")
print(f"Average Trajectory Length: {mean_avg_length:.2f} ± {std_avg_length:.2f} mm")
print(f"Average Time Cost: {mean_avg_timecost:.2f} ± {std_avg_timecost:.2f} steps")

results_dir = f"/home/jwu220/Trajectory_cloud/IL_model_v2/{task_name}/R3M_{view_name}_view/Results"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"{task_name}_{view_name}_results.txt")

with open(results_file, 'w') as f:
    f.write(f"Task: {task_name}\n")
    f.write(f"View: {view_name}\n")
    f.write(f"Number of episodes per experiment: {num_episodes}\n")
    f.write(f"Number of experiments: {num_experiments}\n\n")
    f.write("Results:\n")
    f.write(f"Success Rate: {mean_success_rate:.2%} ± {std_success_rate:.2%}\n")
    f.write(f"Average Trajectory Length: {mean_avg_length:.2f} ± {std_avg_length:.2f} mm\n")
    f.write(f"Average Time Cost: {mean_avg_timecost:.2f} ± {std_avg_timecost:.2f} steps\n")

print(f"\nResults saved to {results_file}")