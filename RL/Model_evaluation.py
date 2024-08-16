import os
import argparse
import numpy as np
import gymnasium as gym
import importlib
from stable_baselines3.common.utils import set_random_seed
from algorithm_configs_online import get_algorithm_config
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
Base_directory = "/home/jwu220/Trajectory_cloud/RL_new"

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
                   max_episode_step=max_episode_steps, seed=args.eval_seed, step_size=step_size, threshold=threshold)
    return env, step_size, threshold, max_episode_steps

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate trained RL models.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm to evaluate')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    parser.add_argument('--trans_error', type=float, required=True, help='Translational error threshold')
    parser.add_argument('--angle_error', type=float, required=True, help='Angular error threshold in degrees')
    parser.add_argument('--eval_seed', type=int, default=42, help='Fixed seed for evaluation')
    return parser.parse_args()

def load_model(algorithm, env, task_name, reward_type, seed):
    model_path = f"{Base_directory}/{task_name}/{algorithm}/{reward_type}/seed_{seed}/final_model.zip"
    algorithm_config = get_algorithm_config(algorithm, env, task_name, reward_type, seed, None, True)
    model_class = algorithm_config['class']
    return model_class.load(model_path, env=env)

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
            action, _ = model.predict(obs, deterministic=True)
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

def save_results(args, results, train_seeds):
    results_dir = f"{Base_directory}/{args.task_name}/{args.algorithm}/{args.reward_type}/evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results to txt file
    txt_file = os.path.join(results_dir, f"{args.task_name}_{args.algorithm}_{args.reward_type}_results.txt")
    with open(txt_file, 'w') as f:
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Reward Type: {args.reward_type}\n")
        f.write(f"Number of seeds: {len(train_seeds)}\n")
        f.write(f"Evaluation seed: {args.eval_seed}\n\n")
        f.write("Results:\n")
        f.write(f"Success Rate: {results['mean_success_rate']:.2%} ± {results['std_success_rate']:.2%}\n")
        f.write(f"Average Trajectory Length: {results['mean_avg_length']:.2f} ± {results['std_avg_length']:.2f} mm\n")
        f.write(f"Average Time Cost: {results['mean_avg_timecost']:.2f} ± {results['std_avg_timecost']:.2f} steps\n")
    
    print(f"\nDetailed results saved to {txt_file}")

    numbers_file = os.path.join(results_dir, f"{args.task_name}_{args.algorithm}_{args.reward_type}_numbers.txt")
    with open(numbers_file, 'w') as f:
        f.write(f"{results['mean_success_rate']} {results['std_success_rate']} ")
        f.write(f"{results['mean_avg_length']} {results['std_avg_length']} ")
        f.write(f"{results['mean_avg_timecost']} {results['std_avg_timecost']}")
    
    print(f"Numeric results saved to {numbers_file}")

def main():
    args = parse_arguments()
    set_random_seed(args.eval_seed)
    
    env, step_size, threshold, max_episode_steps = setup_environment(args)
    
    train_seeds = [1, 10, 100, 1000, 10000]
    all_success_rates = []
    all_lengths = []
    all_timecosts = []
    
    for train_seed in train_seeds:
        model = load_model(args.algorithm, env, args.task_name, args.reward_type, train_seed)
        
        num_episodes = 20
        success_rate, avg_length, avg_timecost, lengths, timecosts = run_evaluation(env, model, num_episodes, max_episode_steps)
        
        all_success_rates.append(success_rate)
        all_lengths.extend(lengths)
        all_timecosts.extend(timecosts)
        
        print(f"\nEvaluation Results for model trained with seed {train_seed}:")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Trajectory Length: {avg_length:.2f} mm")
        print(f"Average Time Cost: {avg_timecost:.2f} steps")
    
    # Calculate mean and standard deviation across all seeds
    mean_success_rate = np.mean(all_success_rates)
    std_success_rate = np.std(all_success_rates)
    mean_avg_length = np.mean(all_lengths)
    std_avg_length = np.std(all_lengths)
    mean_avg_timecost = np.mean(all_timecosts)
    std_avg_timecost = np.std(all_timecosts)
    
    print("\nFinal Results Across All Seeds:")
    print(f"Success Rate: {mean_success_rate:.2%} ± {std_success_rate:.2%}")
    print(f"Average Trajectory Length: {mean_avg_length:.2f} ± {std_avg_length:.2f} mm")
    print(f"Average Time Cost: {mean_avg_timecost:.2f} ± {std_avg_timecost:.2f} steps")
    
    results = {
        'mean_success_rate': mean_success_rate,
        'std_success_rate': std_success_rate,
        'mean_avg_length': mean_avg_length,
        'std_avg_length': std_avg_length,
        'mean_avg_timecost': mean_avg_timecost,
        'std_avg_timecost': std_avg_timecost
    }
    
    save_results(args, results, train_seeds)
    
    env.close()

if __name__ == "__main__":
    main()