import os
import argparse
import pickle
import time
import numpy as np
import gymnasium as gym
import importlib
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from algorithm_configs_online import get_algorithm_config
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
Base_directory = os.path.dirname(os.path.abspath(__file__))

def load_expert_data(task_name):
    expert_data_path = Base_directory + f"/Expert_traj/{task_name}/all_episodes_merged.pkl"
    try:
        with open(expert_data_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Expert data file not found: {expert_data_path}")
        return None

def create_model(args, env, expert_data):
    algorithm_config = get_algorithm_config(args.algorithm, env, args.task_name, args.reward_type, args.seed, expert_data)
    model_class = algorithm_config['class']
    model_params = algorithm_config['params']
    return model_class(**model_params)

def setup_environment(args):
    max_episode_steps = 300
    trans_step = 1.0e-3
    angle_step = np.deg2rad(3)
    jaw_step = 0.05
    step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step], dtype=np.float32)

    threshold = np.array([args.trans_error, np.deg2rad(args.angle_error)], dtype=np.float32)
    
    module_name = f"{args.task_name.capitalize()}_env"
    class_name = f"SRC_{args.task_name.lower()}"
    module = importlib.import_module(module_name)
    SRC_class = getattr(module, class_name)
    
    gym.envs.register(id=f"{args.algorithm}_{args.reward_type}", entry_point=SRC_class, max_episode_steps=max_episode_steps)
    env = gym.make(f"{args.algorithm}_{args.reward_type}", render_mode="human", reward_type=args.reward_type,
                   max_episode_step=max_episode_steps, seed=args.seed, step_size=step_size, threshold=threshold)
    return env, step_size, threshold, max_episode_steps

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm to use')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    parser.add_argument('--total_timesteps', type=int, default=500000, help='Total timesteps for training')
    parser.add_argument('--save_freq', type=int, default=50000, help='Frequency of saving checkpoints')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--trans_error', type=float, required=True, help='Translational error threshold')
    parser.add_argument('--angle_error', type=float, required=True, help='Angular error threshold in degrees')
    return parser.parse_args()

def main():
    args = parse_arguments()
    set_random_seed(args.seed)
    
    # Setup the environment
    env, step_size, threshold, max_episode_steps = setup_environment(args)
    
    # Load expert data
    expert_data = load_expert_data(args.task_name)
    
    # Create the model
    model = create_model(args, env, expert_data)
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=f"{Base_directory}/{args.task_name}/{args.algorithm}/{args.reward_type}/seed_{args.seed}/checkpoints/",
        name_prefix="rl_model"
    )
    
    # Train the model
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=checkpoint_callback, reset_num_timesteps=False)
    
    # Save the final model
    save_path = f"{Base_directory}/{args.task_name}/{args.algorithm}/{args.reward_type}/seed_{args.seed}/final_model"
    model.save(save_path)
    print(f"Final model saved to {save_path}")

    env.close()

if __name__ == "__main__":
    main()