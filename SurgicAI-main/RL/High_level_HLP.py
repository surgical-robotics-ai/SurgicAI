import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
# from EasyEnv import myEasyGym
from High_level_env import SRC_high_level
# from SRC_v5_relative import SRC_test
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import HerReplayBuffer, DDPG, PPO, SAC
from Model_free_Approach.DDPG_BC import DDPG_BC
from Model_free_Approach.td3_BC import TD3_BC
from Model_free_Approach.DemoHerReplayBuffer import DemoHerReplayBuffer
from stable_baselines3.common.utils import set_random_seed
import time
# Create environment

seed = 10
set_random_seed(seed)


episode_steps = 3000

gym.envs.register(id="high_level", entry_point=SRC_high_level, max_episode_steps=episode_steps)
env = gym.make("high_level")


import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

# First training
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="/home/jin/SRC-gym/gym-env/Hierachical_Learning/High_level")
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='/home/jin/SRC-gym/gym-env/Hierachical_Learning/High_level/HLC', name_prefix='HLC')
model.learn(total_timesteps=int(2000000), progress_bar=True,callback=checkpoint_callback,)