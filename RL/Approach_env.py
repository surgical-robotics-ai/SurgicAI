# Training environment for approaching stage for (DDPG+HER)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from PyKDL import Frame, Rotation, Vector
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
        # low_limits = [-0.02, -0.02, -0.01, -np.deg2rad(30), -np.deg2rad(30), -np.deg2rad(30), -0.1]
        # high_limits = [0.02, 0.02, 0.01, np.deg2rad(30), np.deg2rad(30), np.deg2rad(30), 0.1]
        # random_array = np.random.uniform(low=low_limits, high=high_limits)
        # self.psm_goal_list[self.psm_idx-1] = self.init_psm2+random_array
        self.min_angle = 5
        self.max_angle = 20
        self.grasp_angle = np.random.uniform(self.min_angle, self.max_angle)

        self.psm_goal_list[self.psm_idx-1] = np.copy(self.init_psm2)
        self.psm_step(self.psm_goal_list[self.psm_idx-1],self.psm_idx)
        self.world_handle.reset()
        self.Camera_view_reset()
        time.sleep(0.5)

        self.needle_obs = self.needle_goal_evaluator(0.007)
        self.goal_obs = self.needle_obs
        self.multigoal_obs = self.needle_multigoal_evaluator(lift_height=0.007,start_degree=self.min_angle,end_degree=self.max_angle)
        self.init_obs_array = np.concatenate((self.psm_goal_list[self.psm_idx-1],self.goal_obs,self.goal_obs-self.psm_goal_list[self.psm_idx-1]),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":self.psm_goal_list[self.psm_idx-1],"desired_goal":self.goal_obs}

        self.obs = self.normalize_observation(self.init_obs_dict)
        
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        
        return self.obs, self.info
    
    def step(self, action):
        self.needle_obs = self.needle_goal_evaluator(lift_height=0.007,deg_angle=self.grasp_angle)
        self.multigoal_obs = self.needle_multigoal_evaluator(lift_height=0.007,start_degree=self.min_angle,end_degree=self.max_angle)
        self.goal_obs = self.needle_obs
        return super(SRC_approach, self).step(action)

    def Manual_reset(self,init_obs,init_needle):
        self.psm2.actuators[0].deactuate()
        """ Reset the state of the environment to an initial state """

        self.needle.needle.set_pose(init_needle)
        self.psm_goal_list[self.psm_idx-1] = init_obs
        self.psm_step(self.psm_goal_list[self.psm_idx-1],2)
        self.world_handle.reset()
        self.Camera_view_reset()
        time.sleep(0.5)

        self.needle_obs = self.needle_goal_evaluator(0.007)
        self.init_obs_array = np.concatenate((self.init_psm2,self.needle_obs,self.needle_obs-self.init_psm2),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":self.init_psm2,"desired_goal":self.needle_obs}

        self.obs = self.normalize_observation(self.init_obs_dict)
        
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info


    def criteria(self):
        achieved_goal = self.obs["achieved_goal"]

        min_trans = np.Inf
        min_angle = np.Inf
        
        for idx, desired_goal in enumerate(self.multigoal_obs):
            desired_goal = desired_goal*np.array([100,100,100,1,1,1,1])
            distances_trans = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
            distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
            
            if min_trans > distances_trans:
                min_trans = distances_trans
            if min_angle > distances_angle:
                min_angle = distances_angle

            if distances_trans <= self.threshold_trans and distances_angle <= self.threshold_angle and self.jaw_angle_list[self.psm_idx-1] <= 0.1:
                print(f"Matched degree is {self.needle_kin.start_degree + idx * (self.needle_kin.end_degree - self.needle_kin.start_degree) / self.needle_kin.num_points}, distance_trans = {distances_trans}, distances_angle = {np.degrees(distances_angle)}")
                print("Attach the needle to the gripper")
                self.psm2.actuators[0].actuate("Needle")
                self.needle.needle.set_force([0.0,0.0,0.0])
                self.needle.needle.set_torque([0.0,0.0,0.0])
                return True
            
        self.min_trans = min_trans
        self.min_angle = min_angle    
        # print(f"min trans = {min_trans} cm, min_angle = {np.degrees(min_angle)} deg")
        return False
    
    def needle_multigoal_evaluator(self, lift_height=0.007, psm_idx=2, start_degree=5, end_degree=30, num_points=25):
        """
        Evaluate the multiple allowed goal grasping points.
        """
        interpolated_transforms = self.needle_kin.get_interpolated_transforms(start_degree, end_degree, num_points)
        goals = []

        for transform in interpolated_transforms:
            grasp_in_World = transform

            lift_in_grasp_rot = Rotation(1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1)
            lift_in_grasp_trans = Vector(0, 0, lift_height)
            lift_in_grasp = Frame(lift_in_grasp_rot, lift_in_grasp_trans)

            if psm_idx == 2:
                gripper_in_lift_rot = Rotation(0, -1, 0,
                                               -1, 0, 0,
                                               0, 0, -1)
            else:
                gripper_in_lift_rot = Rotation(0, 1, 0,
                                               1, 0, 0,
                                               0, 0, -1)

            gripper_in_lift_trans = Vector(0.0, 0.0, 0.0)
            gripper_in_lift = Frame(gripper_in_lift_rot, gripper_in_lift_trans)

            gripper_in_world = grasp_in_World * lift_in_grasp * gripper_in_lift
            gripper_in_base = self.psm_list[psm_idx - 1].get_T_w_b() * gripper_in_world

            array_goal_base = self.Frame2Vec(gripper_in_base)
            array_goal_base = np.append(array_goal_base, 0.0)
            goals.append(array_goal_base)

        return goals
 

