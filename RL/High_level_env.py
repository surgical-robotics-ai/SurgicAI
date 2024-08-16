# Training environment for approaching stage for (DDPG+HER)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from Low_env_init import low_level_controller
from surgical_robotics_challenge.kinematics.psmFK import *

def add_break(s):
    time.sleep(s)

class SRC_high_level(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human'],"reward_type":['dense']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __init__(self):
        super(SRC_high_level, self).__init__()
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 35, dtype=np.float32),
            high=np.array([np.inf] * 35, dtype=np.float32),
            shape=(35,),
            dtype=np.float32
        )
        self.LLC = low_level_controller()
        self.low_env = self.LLC.env
        self.scale_factor = np.array([100, 100, 100, 1, 1, 1, 1])
        return

    def reset(self, **kwargs):
        self.timer = 0
        self.task_idx_test = 1
        self.obs_low,self.info = self.low_env.reset()

        psm1_pos = self.low_env.psm1.get_T_b_w()*convert_mat_to_frame(self.low_env.psm1.measured_cp())
        psm2_pos = self.low_env.psm2.get_T_b_w()*convert_mat_to_frame(self.low_env.psm2.measured_cp())
        needle_pos = self.low_env.needle_kin.get_bm_pose()
        psm1_jaw = self.low_env.psm_goal_list[0][-1]
        psm2_jaw = self.low_env.psm_goal_list[1][-1]
        psm1_obs = np.concatenate((self.low_env.Frame2Vec(psm1_pos),np.array([psm1_jaw])))*self.scale_factor 
        psm2_obs = np.concatenate((self.low_env.Frame2Vec(psm2_pos),np.array([psm2_jaw])))*self.scale_factor 
        needle_obs = np.concatenate((self.low_env.Frame2Vec(needle_pos),np.array([0.0])))*self.scale_factor
        psm1_needle_world = needle_obs-psm1_obs
        psm2_needle_world = needle_obs-psm2_obs
        self.obs = np.concatenate((psm1_obs,psm2_obs,needle_obs,psm1_needle_world,psm2_needle_world),dtype=np.float32)
        
        return self.obs, self.info
    
    def step(self, action):
        subtimer = 0
        task_idx = action+1
        self.low_env.task_update(task_idx)

        while (self.low_env.subtask_completion == 0):
            next_obs_low, _, self.terminate, self.truncate, self.info = self.LLC.low_level_step(self.obs_low)
            self.obs_low = next_obs_low
            subtimer += 1
            if (subtimer>200):
                break
        
        psm1_pos = self.low_env.psm1.get_T_b_w()*convert_mat_to_frame(self.low_env.psm1.measured_cp())
        psm2_pos = self.low_env.psm2.get_T_b_w()*convert_mat_to_frame(self.low_env.psm2.measured_cp())
        needle_pos = self.low_env.needle_kin.get_bm_pose()
        psm1_jaw = self.low_env.psm_goal_list[0][-1]
        psm2_jaw = self.low_env.psm_goal_list[1][-1]
        psm1_obs = np.concatenate((self.low_env.Frame2Vec(psm1_pos),np.array([psm1_jaw])))*self.scale_factor 
        psm2_obs = np.concatenate((self.low_env.Frame2Vec(psm2_pos),np.array([psm2_jaw])))*self.scale_factor 
        needle_obs = np.concatenate((self.low_env.Frame2Vec(needle_pos),np.array([0.0])))*self.scale_factor
        psm1_needle_world = needle_obs-psm1_obs
        psm2_needle_world = needle_obs-psm2_obs
        self.obs = np.concatenate((psm1_obs,psm2_obs,needle_obs,psm1_needle_world,psm2_needle_world),dtype=np.float32)

        self.reward = float(int(self.terminate)*10)
        self.timer+=1
        return self.obs, self.reward, self.terminate, self.truncate, self.info
    
    def render(self, mode='human', close=False):
        pass
