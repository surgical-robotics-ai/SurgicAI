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
        """ Reset the state of the environment to an initial state """
        self.world_handle.reset()
        self.needle_randomization()
        time.sleep(2.0)
        ######################
        self.psm_goal_list[self.psm_idx-1] = self.needle_random_grasping_evaluator(0.010)
        self.psm_step(self.psm_goal_list[self.psm_idx-1],self.psm_idx)
        self.Camera_view_reset()
        time.sleep(2.0)

        self.psm2.actuators[0].actuate("Needle")
        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])
        # self.entry_obs = self.entry_goal_evaluator(idx=2,noise=True)
        self.entry_obs = self.entry_goal_evaluator(idx=2,dev_trans=[0,0,0.001],noise=False) # Close noise in this case
        self.psm_goal_list[self.psm_idx-1]  = self.entry_obs
        self.psm_step_move(self.psm_goal_list[self.psm_idx-1] ,2,execute_time=0.6)
        print("psm2 move to the goal positions")
        time.sleep(1.0)

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

    def entry_goal_evaluator(self,deg=120,dev_trans=[0,0,0],dev_Yangle = 0.0,idx=2,noise=False):
        rotation_noise = Rotation.RotY(np.deg2rad(dev_Yangle))
        translation_noise = Vector(0, 0, 0)
        noise_in_entry = Frame(rotation_noise, translation_noise)

        entry_in_world = self.scene.entry1_measured_cp()
        noise_in_world = entry_in_world*noise_in_entry
        entry_in_base = self.psm_list[idx-1].get_T_w_b()*noise_in_world # entry with angle deviation

        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_tip_in_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_tip_in_entry = Vector(dev_trans[0],dev_trans[1],dev_trans[2])
        tip_in_entry = Frame(rotation_tip_in_entry,trans_tip_in_entry)


        tip_in_world = self.needle_kin.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_tip = tip_in_world.Inverse()*gripper_in_world

        gripper_in_base = entry_in_base*tip_in_entry*gripper_in_tip
        array_insert = self.Frame2Vec(gripper_in_base)
        array_insert = np.append(array_insert,0.0)
        if noise:
            ranges = np.array([0.001, 0.001, 0.001, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), 0])
            random_noise = np.random.uniform(-ranges, ranges)
            array_insert += random_noise
        return array_insert
    
    def insert_goal_evaluator(self,deg=120,dev=[0,0,0],idx=2):
        exit_in_world = self.scene.exit1_measured_cp()
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*exit_in_world

        # entry_pos.Inverse() to obtain the inverse transformation matrix
        # rotation_matrix = np.array([[0,-1,0],[0,0,1],[-1,0,0]]).astype(np.float32)
        rotation_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]]).astype(np.float32)
        rotation_front_in_exit = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_front_in_exit = Vector(dev[0],dev[1],dev[2])
        front_in_exit = Frame(rotation_front_in_exit,trans_front_in_exit)

        front_in_world = self.needle_kin.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_front = front_in_world.Inverse()*gripper_in_world

        gripper_in_base = exit_in_base*front_in_exit*gripper_in_front
        array_insert = self.Frame2Vec(gripper_in_base)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def needle_random_grasping_evaluator(self,lift_height):
        self.random_degree = np.random.uniform(10, 50)
        self.grasping_pos = self.needle_kin.get_random_grasp_point()
        needle_rot = self.grasping_pos.M
        needle_trans_lift = Vector(self.grasping_pos.p.x(),self.grasping_pos.p.y(),self.grasping_pos.p.z()+lift_height)
        needle_goal_lift = Frame(needle_rot, needle_trans_lift)

        T_calibrate = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_calibrate[:3, :3]

        rotation_calibrate = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        needle_goal_lift.M = needle_goal_lift.M * rotation_calibrate # To be tested
        
        psm_goal_lift = self.psm2.get_T_w_b()*needle_goal_lift

        T_goal = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_goal[:3, :3]

        rotation = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        psm_goal_lift.M = psm_goal_lift.M*rotation

        array_goal_base = self.Frame2Vec(psm_goal_lift)
        array_goal_base = np.append(array_goal_base,0.0)
        return array_goal_base
    
