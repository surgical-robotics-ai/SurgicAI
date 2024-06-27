# Training environment for approaching stage for (DDPG+HER)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import re

from PyKDL import Frame, Rotation, Vector
from gym.spaces.box import Box
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.task_completion_report import TaskCompletionReport
from surgical_robotics_challenge.utils.task3_init import NeedleInitialization
from surgical_robotics_challenge.evaluation.evaluation import Task_2_Evaluation, Task_2_Evaluation_Report
from utils.observation import Observation
from utils.needle_kinematics_v2 import NeedleKinematics_v2
from evaluation import *
from surgical_robotics_challenge.kinematics.psmFK import *
from subtask_env import SRC_subtask


class SRC_approach(SRC_subtask):
    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=None):

        # Define action and observation space
        super(SRC_approach, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 2

    def reset(self,**kwargs):
        
        """ Reset the state of the environment to an initial state """
        self.psm2.actuators[0].deactuate()
        self.needle_randomization()
        ######################
        low_limits = [-0.02, -0.02, -0.01, -np.deg2rad(30), -np.deg2rad(30), -np.deg2rad(30), -0.1]
        high_limits = [0.02, 0.02, 0.01, np.deg2rad(30), np.deg2rad(30), np.deg2rad(30), 0.1]
        random_array = np.random.uniform(low=low_limits, high=high_limits)
        self.psm_goal_list[self.psm_idx-1] = self.init_psm2+random_array

        # self.psm_goal_list[self.psm_idx-1] = self.init_psm2
        self.psm_step(self.psm_goal_list[self.psm_idx-1],self.psm_idx)
        self.world_handle.reset()
        self.Camera_view_reset()
        time.sleep(0.5)

        self.needle_obs = self.needle_goal_evaluator(0.010)
        self.goal_obs = self.needle_obs
        self.init_obs_array = np.concatenate((self.psm_goal_list[self.psm_idx-1],self.goal_obs,self.goal_obs-self.psm_goal_list[self.psm_idx-1]),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":self.psm_goal_list[self.psm_idx-1],"desired_goal":self.goal_obs}

        self.obs = self.normalize_observation(self.init_obs_dict)
        
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        
        return self.obs, self.info
    
    def Manual_reset(self,init_obs,init_needle):
        self.psm2.actuators[0].deactuate()
        """ Reset the state of the environment to an initial state """

        self.needle.needle.set_pose(init_needle)
        ######################

        self.psm_goal_list[self.psm_idx-1] = init_obs
        self.psm_step(self.psm_goal_list[self.psm_idx-1],2)
        self.world_handle.reset()
        self.Camera_view_reset()
        time.sleep(0.5)

        self.needle_obs = self.needle_goal_evaluator(0.010)
        self.init_obs_array = np.concatenate((self.init_psm2,self.needle_obs,self.needle_obs-self.init_psm2),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":self.init_psm2,"desired_goal":self.needle_obs}

        self.obs = self.normalize_observation(self.init_obs_dict)
        
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info

    def criteria(self):
        """
        Decide whether success criteria (Distance is lower than a threshold) is met.
        """
        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        distances_trans = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
        jaw_angle = self.obs["achieved_goal"][-1]
        if (distances_trans<= self.threshold_trans) and (distances_angle <= self.threshold_angle and jaw_angle<=0.1):
            return True
        else:
            return False


 

