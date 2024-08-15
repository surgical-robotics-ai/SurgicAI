from typing import Dict, Any
from RL_algo.PPO import PPO
from RL_algo.DDPG import DDPG
from RL_algo.SAC import SAC
from RL_algo.td3_BC import TD3_BC
from RL_algo.td3 import TD3
from RL_algo.BC import BC
from RL_algo.DemoHerReplayBuffer import DemoHerReplayBuffer
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import os

# Constants
DEFAULT_NET_ARCH = dict(pi=[256, 256, 256], qf=[256, 256, 256])
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 2e-4

def get_algorithm_config(algorithm_name: str, env: Any, task_name: str, reward_type: str, seed: int, episode_transitions: Any = None, evaluation_mode: bool = False) -> Dict[str, Any]:
    """
    Get the configuration for the specified algorithm.

    :param algorithm_name: Name of the algorithm
    :param env: The environment
    :param task_name: Name of the task
    :param reward_type: Type of reward (dense or sparse)
    :param seed: Random seed
    :param episode_transitions: Expert demonstrations for BC-based algorithms
    :return: Dictionary containing the algorithm class and its parameters
    """
    base_path = f"/home/robo/trajectory_data/RL_new/{task_name}/{algorithm_name}/{reward_type}/seed_{seed}"
    os.makedirs(base_path, exist_ok=True)
    
    base_params = {
        "policy": "MultiInputPolicy",
        "env": env,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "gamma": 0.995,
        "batch_size": DEFAULT_BATCH_SIZE,
        "verbose": 1,
        "seed": seed,
    }

    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=5e-2 * np.ones(env.action_space.shape[-1]))

    configs = {
        "PPO": {
            "class": PPO,
            "params": {
                **base_params,
                "policy_kwargs": dict(net_arch=DEFAULT_NET_ARCH),
                "tensorboard_log": f"{base_path}/tensorboard",
            }
        },
        "TD3_HER_BC": {
            "class": TD3_BC,
            "params": {
                **base_params,
                "tau": 0.005,
                "action_noise": action_noise,
                "replay_buffer_class": DemoHerReplayBuffer,
                "train_freq": (2, "episode"),
                "policy_kwargs": dict(net_arch=DEFAULT_NET_ARCH),
                "replay_buffer_kwargs": dict(
                    demo_transitions=episode_transitions,
                    demo_sample_ratio=0.3,
                    n_sampled_goal=4,
                    goal_selection_strategy="future",
                ),
                "tensorboard_log": f"{base_path}/tensorboard",
                "episode_transitions": episode_transitions,
                "BC_coeff": 0.6,
                "demo_ratio": 0.3,
            }
        },
        "DDPG_HER": {
            "class": DDPG,
            "params": {
                **base_params,
                "tau": 0.005,
                "replay_buffer_class": HerReplayBuffer,
                "train_freq": (2, "episode"),
                "policy_kwargs": dict(net_arch=DEFAULT_NET_ARCH),
                "replay_buffer_kwargs": dict(
                    n_sampled_goal=4,
                    goal_selection_strategy="future",
                ),
                "tensorboard_log": f"{base_path}/tensorboard",
            }
        },
        "BC": {
            "class": BC,
            "params": {
                "policy": "MultiInputPolicy",
                "env": env,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "batch_size": DEFAULT_BATCH_SIZE,
                "verbose": 1,
                "seed": seed,
                "policy_kwargs": dict(net_arch=DEFAULT_NET_ARCH),
                "tensorboard_log": f"{base_path}/tensorboard",
                "episode_transitions": episode_transitions,
            }
        },
        "DDPG": {
            "class": DDPG,
            "params": {
                **base_params,
                "action_noise": action_noise,
                "tau": 0.005,
                "train_freq": (2, "episode"),
                "tensorboard_log": f"{base_path}/tensorboard",
            }
        },
        "SAC": {
            "class": SAC,
            "params": {
                **base_params,
                "tau": 0.005,
                "train_freq": (2, "episode"),
                "policy_kwargs": dict(net_arch=DEFAULT_NET_ARCH),
                "tensorboard_log": f"{base_path}/tensorboard",
            }
        },
        "TD3": {
            "class": TD3,
            "params": {
                **base_params,
                "tau": 0.005,
                "action_noise": action_noise,
                "train_freq": (2, "episode"),
                "policy_kwargs": dict(net_arch=DEFAULT_NET_ARCH),
                "tensorboard_log": f"{base_path}/tensorboard",
            }
        },
        "TD3_HER": {
            "class": TD3,
            "params": {
                **base_params,
                "tau": 0.005,
                "replay_buffer_class": HerReplayBuffer,
                "train_freq": (2, "episode"),
                "policy_kwargs": dict(net_arch=DEFAULT_NET_ARCH),
                "replay_buffer_kwargs": dict(
                    n_sampled_goal=4,
                    goal_selection_strategy="future",
                ),
                "tensorboard_log": f"{base_path}/tensorboard",
            }
        },
    }

    if algorithm_name not in configs:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    config = configs[algorithm_name]
    
    if "episode_transitions" in config["params"] and episode_transitions is None and not evaluation_mode:
        raise ValueError(f"{algorithm_name} requires episode_transitions, but none were provided.")

    return config