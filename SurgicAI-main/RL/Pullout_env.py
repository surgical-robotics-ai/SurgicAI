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
import random
from subtask_env import SRC_subtask

class SRC_pullout(SRC_subtask):

    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=np.array([0.0005, 0.0005, 0.0005, np.deg2rad(2), np.deg2rad(2), np.deg2rad(2), 0.05])):

        super(SRC_pullout, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 1

    def reset(self, seed = None,**kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.psm1.actuators[0].deactuate()
        self.psm_goal_list[self.psm_idx-1] = self.init_psm1
        self.psm_step(self.psm_goal_list[self.psm_idx-1],self.psm_idx)
        """ Reset the state of the environment to an initial state """
        self.world_handle.reset()
        self.needle_initialization()
        self.Camera_view_reset()
        time.sleep(0.5)
        self.psm1_goal = self.needle_goal_evaluator(deg_angle=105,psm_idx=1,lift_height=0.007)
        self.psm_step(np.concatenate((self.psm1_goal[0:-1],[0.8])), self.psm_idx)
        time.sleep(0.5)
        self.psm1.set_jaw_angle(0.0)
        time.sleep(0.5)
        self.psm1.actuators[0].actuate("Needle")
        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])
        time.sleep(0.5)
        
        self.goal_obs = self.handover_goal_evaluator(idx=1)
        self.psm_goal_list[self.psm_idx-1] = self.psm1_goal
        obs_array = np.concatenate((self.psm1_goal,self.goal_obs,self.goal_obs-self.psm1_goal), dtype=np.float32)
        
        self.init_obs_dict = {"observation":obs_array,"achieved_goal":self.psm1_goal,"desired_goal":self.goal_obs}
        self.obs = self.normalize_observation(self.init_obs_dict) 

        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info
    
    def handover_goal_evaluator(self,deg=110,dev=[0,0,0],idx=1):
        exit_in_world = self.scene.exit1_measured_cp()
        rotation_decrease_y = Rotation.RotY(-np.deg2rad(50))
        new_rotation = exit_in_world.M * rotation_decrease_y
        handover_in_world = handover_in_world = Frame(new_rotation, Vector(exit_in_world.p[0] + 0.03, exit_in_world.p[1], exit_in_world.p[2] + 0.03))
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*handover_in_world

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
        array_handover = self.Frame2Vec(gripper_in_base)
        array_handover = np.append(array_handover,0.0)
        return array_handover
   
    def entry_needle_pose(self,deg=120,dev_trans=[0,0,0],dev_Yangle = 0.0,noise=False):
        rotation_noise = Rotation.RotY(np.deg2rad(dev_Yangle))
        translation_noise = Vector(0, 0, 0)
        entry_in_old_entry = Frame(rotation_noise, translation_noise)

        old_entry_in_world = self.scene.entry1_measured_cp()
        entry_in_world = old_entry_in_world*entry_in_old_entry 

        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_tip_in_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_tip_in_entry = Vector(dev_trans[0],dev_trans[1],dev_trans[2])
        tip_in_entry = Frame(rotation_tip_in_entry,trans_tip_in_entry)

        tip_in_world = self.needle_kin.get_pose_angle(deg)
        needle_in_world = self.needle_kin.get_pose()
        needle_in_tip = tip_in_world.Inverse()*needle_in_world
        needle_in_world = entry_in_world*tip_in_entry*needle_in_tip

        return needle_in_world
    
    def exit_needle_pose(self,deg=120,dev=[0,0,0],dev_Yangle = 0.0):
        rotation_noise = Rotation.RotY(np.deg2rad(dev_Yangle))
        translation_noise = Vector(0, 0, 0)
        exit_in_old_exit = Frame(rotation_noise, translation_noise)
        old_exit_in_world = self.scene.exit1_measured_cp()
        exit_in_world = old_exit_in_world*exit_in_old_exit 
        exit_in_world = self.scene.exit1_measured_cp()

        rotation_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]]).astype(np.float32)
        rotation_front_in_exit = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        
        trans_front_in_exit = Vector(dev[0],dev[1],dev[2])
        front_in_exit = Frame(rotation_front_in_exit,trans_front_in_exit)

        front_in_world = self.needle_kin.get_pose_angle(deg)
        needle_in_world = self.needle_kin.get_pose()
        needle_in_front = front_in_world.Inverse()*needle_in_world

        needle_in_world = exit_in_world*front_in_exit*needle_in_front
        return needle_in_world

    def needle_initialization(self):
        origin_p = Vector( -0.0207937, 0.0562045, 0.0711726)
        origin_rz = 0.0

        new_rot = Rotation(np.cos(origin_rz),-np.sin(origin_rz),0,
                            np.sin(origin_rz),np.cos(origin_rz),0,
                            0.0,0.0,1.0)
        
        needle_pos_new = Frame(new_rot,origin_p)
        self.needle.needle.set_pose(needle_pos_new)
        time.sleep(1.0)

        ### New version
        self.needle.needle.set_pose(self.entry_needle_pose())
        time.sleep(0.5)
        self.needle.needle.set_pose(self.entry_needle_pose(95,[0.001,0,0],0))
        time.sleep(0.5)
        self.needle.needle.set_pose(self.exit_needle_pose(80,[0.002,0,0],-10))
        time.sleep(0.5)

        random_offset_param1 = random.uniform(-10, 10)  # 对于param1，范围是[-10, 10]
        random_offset_param2 = [random.uniform(-0.001, 0.001) for _ in range(3)]  # 对于param2的每个元素，范围是[-0.001, 0.001]
        random_offset_param3 = random.uniform(-10, 10)  # 对于param3，范围是[-10, 10]

        base_param1 = 100
        base_param2 = [0.001,0,0]
        base_param3 = 10
        final_param1 = base_param1 + random_offset_param1
        final_param2 = [base_param2[i] + random_offset_param2[i] for i in range(3)]
        final_param3 = base_param3 + random_offset_param3
        self.needle.needle.set_pose(self.exit_needle_pose(final_param1,final_param2,final_param3))
        time.sleep(0.5)

    def exit_goal_evaluator(self,deg=110,dev=[0,0,0],idx=1):
        exit_in_world = self.scene.exit1_measured_cp()
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*exit_in_world

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
        array_exit = self.Frame2Vec(gripper_in_base)
        array_exit = np.append(array_exit,0.0)
        return array_exit
