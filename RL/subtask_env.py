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


def add_break(s):
    time.sleep(s)
    # print('-------------')

class SRC_subtask(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human'],"reward_type":['dense']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=None):

        # Define action and observation space
        super(SRC_subtask, self).__init__()

        self.max_timestep = max_episode_step
        print(f"max episode length is {self.max_timestep}")
        self.step_size = step_size
        print(f"step size is {self.step_size}")

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed  # Store the seed if you need to reference it later
            print("Set random seed")
        else:
            self.seed = None  # No seed was provided

        print(f"reward type is {reward_type}")
        self.reward_type = reward_type
        self.threshold_trans = threshold[0]
        self.threshold_angle = threshold[1]
        print(f"Translation threshold: {self.threshold_trans}, angle threshold: {self.threshold_angle}")

        self.action_space = spaces.Box(
                low = np.array([-1,-1,-1,-1,-1,-1,-1],dtype=np.float32),
                high = np.array([1,1,1,1,1,1,1],dtype=np.float32),
                shape = (7,), dtype=np.float32
            )

        self.observation_space = spaces.Dict(
            {
                "observation":spaces.Box(
                    low = np.array([-np.inf] * 21, dtype=np.float32),
                    high = np.array([np.inf] * 21, dtype=np.float32),
                    shape=(21,),dtype=np.float32),
                "achieved_goal": spaces.Box(
                    low=np.array([-np.inf]*7,dtype=np.float32), 
                    high=np.array([np.inf]*7,dtype=np.float32), 
                    shape=(7,),dtype=np.float32),
                "desired_goal": spaces.Box(
                    low=np.array([-np.inf]*7,dtype=np.float32), 
                    high=np.array([np.inf]*7,dtype=np.float32), 
                    shape=(7,),dtype=np.float32)              
            }
        )

        self.simulation_manager = SimulationManager('src_client')
        self.world_handle = self.simulation_manager.get_world_handle()
        self.scene = Scene(self.simulation_manager)
        self.psm1 = PSM(self.simulation_manager, 'psm1',add_joint_errors=False)
        self.psm2 = PSM(self.simulation_manager, 'psm2',add_joint_errors=False)
        self.psm_list = [self.psm1, self.psm2]
        self.ecm = ECM(self.simulation_manager, 'CameraFrame')
        self.ecm2 = ECM(self.simulation_manager, 'FreeCamera')
        self.Camera_view_reset()

        self.needle = NeedleInitialization(self.simulation_manager) # needle obj
        self.needle_kin = NeedleKinematics_v2() # needle movement and positioning
        self.init_psm1 = np.array([ 0.04629208,0.00752399,-0.08173992,-3.598019,-0.05762508,1.2738742,0.8],dtype=np.float32)
        self.psm1_goal = self.init_psm1
        self.psm_step(self.psm1_goal,1)
        add_break(0.5)
        self.init_psm2 = np.array([-0.03721037,  0.01213105, -0.08036895, -2.7039163, 0.07693613, 2.0361109, 0.8],dtype=np.float32)
        self.psm2_goal = self.init_psm2
        self.psm_step(self.psm2_goal,2)
        add_break(0.5)
        self.timestep = 0
        self.psm_idx = None
        self.psm_goal_list = [self.psm1_goal,self.psm2_goal]
        self.goal_obs = None
        print("Initialized!!!")
        return

    def step(self, action):
        """
        Step function, defines the system dynamic and updates the observation
        """
        self.timestep += 1
        current = self.psm_goal_list[self.psm_idx-1]
        # action[-1] = 0
        action_step = action*self.step_size
        self.psm_goal_list[self.psm_idx-1] = current+action_step
        self.world_handle.update()
        self.psm_step(self.psm_goal_list[self.psm_idx-1] ,self.psm_idx)
        self._update_observation(self.psm_goal_list[self.psm_idx-1])
        return self.obs, self.reward, self.terminate, self.truncate, self.info

    def render(self, mode='human', close=False):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """
        Define the reward function for the task
        """
        goal_len = 7
        achieved_goal = np.array(achieved_goal).reshape(-1,goal_len)
        desired_goal = np.array(desired_goal).reshape(-1,goal_len)
        
        distances_trans = np.linalg.norm(achieved_goal[:, 0:3] - desired_goal[:, 0:3], axis=1)
        distances_angle = np.linalg.norm(achieved_goal[:, 3:6] - desired_goal[:, 3:6], axis=1)
        
        assert (self.reward_type == "dense" or self.reward_type == "sparse"), "Wrong reward type"

        if self.reward_type == "dense":
            rewards = -(distances_trans/100+distances_angle/10)
        else:
            rewards = np.where(
                (distances_trans <= self.threshold_trans) & (distances_angle <= self.threshold_angle),
                0,  
                -1  
            )
        return rewards

    def criteria(self):
        """
        Decide whether success criteria (Distance is lower than a threshold) is met.
        """
        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        distances_trans = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
        if (distances_trans<= self.threshold_trans) and (distances_angle <= self.threshold_angle):
            return True
        else:
            return False

    def Camera_view_reset(self):
        rotation = Rotation(-1, 0.0, 0.0,
            0.0, -0.766044, 0.642788,
            0.0, 0.642788, 0.766044)

        vector = Vector(0.0, 0.1463076, 0.187126)
        self.ecm_pos_origin = Frame(rotation, vector)
        self.ecm.servo_cp(self.ecm_pos_origin)

    def normalize_observation(self,observation_dict):
        '''
        Scaling the observations, translation into 'cm', orientation into 'rad'.
        '''
        observation = observation_dict["observation"]
        achieved_goal = observation_dict["achieved_goal"]
        desired_goal = observation_dict["desired_goal"]

        multiplier = np.ones(21, dtype=np.float32)
        indices_to_multiply_100 = [0, 1, 2, 7, 8, 9, 14, 15, 16]
        multiplier[indices_to_multiply_100] = 100
        observation_dict["observation"] = np.array(observation * multiplier,dtype=np.float32)

        multiplier2 = np.array([100,100,100,1,1,1,1])
        observation_dict["achieved_goal"] = np.array(achieved_goal*multiplier2,dtype=np.float32)
        observation_dict["desired_goal"] = np.array(desired_goal*multiplier2,dtype=np.float32)
        
        return observation_dict


    def _update_observation(self, current):
        """ Update the observation of the environment

        Parameters
        - action: an action provided by the environment
        """
        assert self.goal_obs is not None, "Goal_obs is not defined"
        goal_obs = self.goal_obs
        
        current = np.array(current,dtype=np.float32)
        goal_obs = np.array(goal_obs,dtype=np.float32)

        obs_array = np.concatenate((current,goal_obs,goal_obs-current), dtype=np.float32)
        obs_dict = {"observation":obs_array,"achieved_goal":current,"desired_goal":goal_obs}
        self.obs = self.normalize_observation(obs_dict) 

        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        self.reward = self.compute_reward(achieved_goal ,desired_goal) # Already normalized input
        self.reward = float(self.reward)
        self.terminate = False
        self.truncate = False
        self.info = {"is_success":False}
        if self.criteria():
            self.terminate = True
            self.info = {"is_success":True}
            print("Approach the target")
        if self.timestep == self.max_timestep:
            print("Maximum step reaches")
         
    def psm_step(self,obs,psm_idx:int):
        """
        Given obs = np.array([x,y,z,roll,pitch,yaw,jaw]) 
        move the designated psm to the positions.
        """
        X= obs[0]
        Y = obs[1]
        Z = obs[2]
        Roll = obs[3]
        Pitch = obs[4]
        Yaw = obs[5]
        Jaw_angle = obs[6]
        T_goal = Frame(Rotation.RPY(Roll,Pitch,Yaw),Vector(X,Y,Z))
        self.psm_list[psm_idx-1].servo_cp(T_goal)
        self.psm_list[psm_idx-1].set_jaw_angle(Jaw_angle)  


    def psm_step_move(self,obs,psm_idx:int,execute_time=0.5):
        """
        Given obs = np.array([x,y,z,roll,pitch,yaw,jaw]) 
        move the designated psm to the positions.
        """
        X= obs[0]
        Y = obs[1]
        Z = obs[2]
        Roll = obs[3]
        Pitch = obs[4]
        Yaw = obs[5]
        Jaw_angle = obs[6]
        T_goal = Frame(Rotation.RPY(Roll,Pitch,Yaw),Vector(X,Y,Z))
        self.psm_list[psm_idx-1].move_cp(T_goal,execute_time)
        self.psm_list[psm_idx-1].set_jaw_angle(Jaw_angle)

    def needle_goal_evaluator(self,lift_height=0.010, psm_idx=2, deg_angle = None):
        '''
        Evaluate the target goal for needle grasping in Robot frame.
        '''

        if deg_angle is None:
            grasp_in_World = self.needle_kin.get_bm_pose()

        else:
            grasp_in_World = self.needle_kin.get_pose_angle(deg_angle)

        lift_in_grasp_rot = Rotation(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)    
        lift_in_grasp_trans = Vector(0,0,lift_height)
        lift_in_grasp = Frame(lift_in_grasp_rot,lift_in_grasp_trans)

        if psm_idx == 2:
            gripper_in_lift_rot = Rotation(0, -1, 0,
                                            -1, 0, 0,
                                            0, 0, -1)
        else:
            gripper_in_lift_rot = Rotation(0, 1, 0,
                                            1, 0, 0,
                                            0, 0, -1)           

        gripper_in_lift_trans = Vector(0.0,0.0,0.0)
        gripper_in_lift = Frame(gripper_in_lift_rot,gripper_in_lift_trans)

        gripper_in_world = grasp_in_World*lift_in_grasp*gripper_in_lift
        gripper_in_base = self.psm_list[psm_idx-1].get_T_w_b()*gripper_in_world
        

        array_goal_base = self.Frame2Vec(gripper_in_base)
        array_goal_base = np.append(array_goal_base,0.0)
        return array_goal_base


    def needle_randomization(self):
        """
        Initializa needle at random positions in the world
        """
        origin_p = Vector( -0.0207937, 0.0562045, 0.0711726)
        origin_rz = 0.0

        random_x = np.random.uniform(-0.003, 0.003)
        random_y = np.random.uniform(-0.02, 0.01)
        random_rz = np.random.uniform(-np.pi/6,np.pi/6)

        origin_p[0] += random_x
        origin_p[1] += random_y
        origin_rz += random_rz

        new_rot = Rotation(np.cos(origin_rz),-np.sin(origin_rz),0,
                            np.sin(origin_rz),np.cos(origin_rz),0,
                            0.0,0.0,1.0)
        
        needle_pos_new = Frame(new_rot,origin_p)
        self.needle.needle.set_pose(needle_pos_new)
    
    def Frame2Vec(self,goal_frame,bound = True):
        """
        Convert Frame variables into vector forms.
        """
        X_goal = goal_frame.p.x()
        Y_goal = goal_frame.p.y()
        Z_goal = goal_frame.p.z()
        rot_goal = goal_frame.M
        roll_goal,pitch_goal,yaw_goal  = rot_goal.GetRPY()
        if bound:
            if (roll_goal <= np.deg2rad(-360)):
                roll_goal += 2*np.pi
            elif (roll_goal > np.deg2rad(0)):
                roll_goal -= 2*np.pi
        array_goal = np.array([X_goal,Y_goal,Z_goal,roll_goal,pitch_goal,yaw_goal],dtype=np.float32)
        return array_goal
            
