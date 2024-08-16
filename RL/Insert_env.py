import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from PyKDL import Frame, Rotation, Vector
from surgical_robotics_challenge.kinematics.psmFK import *
from subtask_env import SRC_subtask


class SRC_insert(SRC_subtask):
    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=None):

        # Define action and observation space
        super(SRC_insert, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 2
        self.action_lims_low = np.array([-0.1, -0.1, -0.25, np.deg2rad(-270), np.deg2rad(-80), np.deg2rad(-260), 0],dtype=np.float32)
        self.action_lims_high = np.array([0.1, 0.1, 0.05, np.deg2rad(-90), np.deg2rad(80), np.deg2rad(260), 1],dtype=np.float32)

    def reset(self, seed = None,**kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.psm2.actuators[0].deactuate()
        self.psm_goal_list[0] = self.init_psm1
        self.psm_goal_list[1] = self.init_psm2
        self.psm_step(self.psm_goal_list[0],1)
        self.psm_step(self.psm_goal_list[1],2)
        self.world_handle.reset()
        self.Camera_view_reset()
        time.sleep(0.5)
        self.world_handle.reset()
        time.sleep(0.5)

        # Approach and grasp the needle
        self.needle_obs = self.needle_random_grasping_evaluator(0.007)
        self.needle_obs = np.append(self.needle_obs,0.8)
        self.psm_step_move(self.needle_obs,2)
        time.sleep(0.6)
        self.needle_obs[-1] = 0.0
        self.psm_step(self.needle_obs,2)
        self.Camera_view_reset()
        time.sleep(0.3)
        self.psm2.actuators[0].actuate("Needle")
        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])

        # Place the needle at the entry
        self.entry_obs = self.entry_goal_evaluator(idx=2,dev_trans=[0,0,0.001],noise=False) # Close noise in this case
        self.psm_step_move(self.entry_obs,2,execute_time=0.5) 
        time.sleep(0.6)

        self.psm_goal_list[self.psm_idx-1]  = self.entry_obs

        self.goal_obs = self.insert_goal_evaluator(100,[0.002,0,0],self.psm_idx)
        obs_array = np.concatenate((self.psm_goal_list[self.psm_idx-1] ,self.goal_obs,self.goal_obs-self.psm_goal_list[self.psm_idx-1] ), dtype=np.float32)
        
        self.init_obs_dict = {"observation":obs_array,"achieved_goal":self.psm_goal_list[self.psm_idx-1] ,"desired_goal":self.goal_obs}
        self.obs = self.normalize_observation(self.init_obs_dict) 
        
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info
   
    def step(self, action):
        action[-1]=0
        self.timestep += 1
        self.action = action
        current = self.psm_goal_list[self.psm_idx-1]
        action_step = action*self.step_size

        self.psm_goal_list[self.psm_idx-1] = np.clip(current+action_step, self.action_lims_low[0:7], self.action_lims_high[0:7])
        self.world_handle.update()
        self.psm_step(self.psm_goal_list[self.psm_idx-1] ,self.psm_idx)
        self._update_observation(self.psm_goal_list[self.psm_idx-1])
        return self.obs, self.reward, self.terminate, self.truncate, self.info

