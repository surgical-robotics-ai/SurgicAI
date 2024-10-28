import os
import argparse
import pickle
import numpy as np
import gymnasium as gym
import Offline_RL_algo.d3rlpy as d3rlpy
from Offline_RL_algo.d3rlpy.dataset import MDPDataset
from Offline_RL_algo.d3rlpy.metrics.evaluators import EnvironmentEvaluator_dict
from algorithm_configs_offline import get_algorithm_config
import torch
import gc
import importlib

gc.collect()
torch.cuda.empty_cache()
Base_directory = os.path.dirname(os.path.abspath(__file__))
MAX_EPISODE_STEPS = 200

def load_expert_data(task_name):
    expert_data_path = Base_directory + f"/Expert_traj/{task_name}/all_episodes_merged.pkl"
    try:
        with open(expert_data_path, 'rb') as file:
            data = pickle.load(file)
            observations = []
            actions = []
            rewards = []
            terminals = []
            for episode in data:
                observations.extend([episode['obs']['observation']])
                actions.extend([episode['action']])
                rewards.extend([episode['reward']])
                terminals.extend([episode['done']])
            observations = np.array(observations, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            terminals = np.array(terminals, dtype=bool)
            return MDPDataset(observations, actions, rewards, terminals)
    except FileNotFoundError:
        print(f"Expert data file not found: {expert_data_path}")
        return None

def setup_environment(args):
    trans_step = 1.0e-3
    angle_step = np.deg2rad(3)
    jaw_step = 0.05
    step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step], dtype=np.float32)
    threshold = np.array([args.trans_error, np.deg2rad(args.angle_error)], dtype=np.float32)
    module_name = f"{args.task_name.capitalize()}_env"
    class_name = f"SRC_{args.task_name.lower()}"
    module = importlib.import_module(module_name)
    SRC_class = getattr(module, class_name)
    gym.envs.register(id=f"{args.algorithm}_{args.reward_type}", entry_point=SRC_class, max_episode_steps=MAX_EPISODE_STEPS)
    env = gym.make(f"{args.algorithm}_{args.reward_type}", render_mode="human", reward_type=args.reward_type,
                   max_episode_step=MAX_EPISODE_STEPS, seed=args.seed, step_size=step_size, threshold=threshold)
    return env

def run_evaluation(env, model, num_episodes, max_episode_steps):
    total_length = 0
    total_timecost = 0
    total_success = 0
    all_lengths = []
    all_timecosts = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        trajectory_length = 0
        for timestep in range(max_episode_steps):
            action = model.predict(np.array([obs['observation']]))[0]
            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory_length += np.linalg.norm(action[0:3] * env.step_size[0:3] * 1000)
            obs = next_obs
            if terminated:
                total_success += 1
                total_length += trajectory_length
                total_timecost += timestep + 1
                all_lengths.append(trajectory_length)
                all_timecosts.append(timestep + 1)
                break
    
    success_rate = total_success / num_episodes
    avg_length = total_length / total_success if total_success > 0 else 0
    avg_timecost = total_timecost / total_success if total_success > 0 else 0
    
    return success_rate, avg_length, avg_timecost, all_lengths, all_timecosts

def save_results(args, results):
    success_rate, avg_length, avg_timecost, all_lengths, all_timecosts = results
    
    mean_success_rate = success_rate
    std_success_rate = 0  # Since it's a single success rate over multiple episodes
    
    mean_avg_length = np.mean(all_lengths)
    std_avg_length = np.std(all_lengths)
    
    mean_avg_timecost = np.mean(all_timecosts)
    std_avg_timecost = np.std(all_timecosts)
    
    results_dir = f"{Base_directory}/{args.task_name}/{args.algorithm}/{args.reward_type}/seed_{args.seed}/evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results to txt file
    txt_file = os.path.join(results_dir, f"{args.task_name}_{args.algorithm}_{args.reward_type}_results.txt")
    with open(txt_file, 'w') as f:
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Reward Type: {args.reward_type}\n")
        f.write(f"Evaluation seed: {args.seed}\n\n")
        f.write("Results:\n")
        f.write(f"Success Rate: {mean_success_rate:.2%}\n")
        f.write(f"Average Trajectory Length: {mean_avg_length:.2f} ± {std_avg_length:.2f} mm\n")
        f.write(f"Average Time Cost: {mean_avg_timecost:.2f} ± {std_avg_timecost:.2f} steps\n")
    
    print(f"\nDetailed results saved to {txt_file}")

    # Save numeric results to txt file
    numbers_file = os.path.join(results_dir, f"{args.task_name}_{args.algorithm}_{args.reward_type}_numbers.txt")
    with open(numbers_file, 'w') as f:
        f.write(f"{mean_success_rate} {std_success_rate} ")
        f.write(f"{mean_avg_length} {std_avg_length} ")
        f.write(f"{mean_avg_timecost} {std_avg_timecost}")
    
    print(f"Numeric results saved to {numbers_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an offline RL agent.")
    parser.add_argument('--algorithm', type=str, required=True, choices=['CQL', 'CalQL', 'IQL', 'BCQ', 'AWAC'], help='Name of the offline RL algorithm to use')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--n_steps_per_epoch', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--trans_error', type=float, required=True, help='Translational error threshold')
    parser.add_argument('--angle_error', type=float, required=True, help='Angular error threshold in degrees')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d3rlpy.seed(args.seed)

    env = setup_environment(args)
    dataset = load_expert_data(args.task_name)
    if dataset is None:
        print("No expert data found. Exiting.")
        return

    model = get_algorithm_config(args.algorithm, env, args.task_name, args.reward_type, args.seed, args.use_gpu)

    model.fit(
        dataset,
        n_steps=args.n_epochs * args.n_steps_per_epoch,
        n_steps_per_epoch=args.n_steps_per_epoch,
        experiment_name=f"{args.task_name}_{args.algorithm}_{args.reward_type}",
        with_timestamp=False,
        save_interval=1,
        evaluators={"environment": EnvironmentEvaluator_dict(env)},
        show_progress=True,
    )

    save_path = f"{Base_directory}/{args.task_name}/{args.algorithm}/{args.reward_type}/seed_{args.seed}/final_model.d3"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    print("\nStarting model evaluation...")
    num_episodes = 20
    evaluation_results = run_evaluation(env, model, num_episodes, MAX_EPISODE_STEPS)
    
    save_results(args, evaluation_results)
    
    env.close()

if __name__ == "__main__":
    main()