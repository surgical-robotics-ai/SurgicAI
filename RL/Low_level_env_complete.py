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
from surgical_robotics_challenge.utils.task3_init import NeedleInitialization
from utils.needle_kinematics_old import NeedleKinematics_v2
from surgical_robotics_challenge.kinematics.psmFK import *


def add_break(s):
    time.sleep(s)
    # print('-------------')

class SRC_low_level(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human'],"reward_type":['dense']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = None,max_episode_step=800, step_size=None):

        # Define action and observation space
        super(SRC_low_level, self).__init__()

        self.max_timestep = max_episode_step
        
        print(f"max episode length is {self.max_timestep}")
        self.step_size = step_size
        # print(f"step size is {self.step_size}")

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed  # Store the seed if you need to reference it later
            print("Seed set")
        else:
            self.seed = None  # No seed was provided

        print(f"reward type is {reward_type}")
        self.reward_type = reward_type
        self.threshold = threshold
        print(f"Grasping Translation threshold: {self.threshold[0]}, angle threshold: {self.threshold[1]}")
        print(f"Inserting Translation threshold: {self.threshold[2]}, angle threshold: {self.threshold[3]}")
        print(f"Passing Translation threshold: {self.threshold[4]}, angle threshold: {self.threshold[5]}")
        print(f"Regrasping Translation threshold: {self.threshold[6]}, angle threshold: {self.threshold[7]}")
        print(f"Handover Translation threshold: {self.threshold[8]}, angle threshold: {self.threshold[9]}")

        # Limits for psm
        self.action_space = spaces.Box(
                low = np.array([-1,-1,-1,-1,-1,-1,-1],dtype=np.float32),
                high = np.array([1,1,1,1,1,1,1],dtype=np.float32),
                shape = (7,), dtype=np.float32
            )
        # Add three observation variable at the end: x,y,rz
        # Boundaries are defined in normalize_observation
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

        # Connect to client using SimulationManager
        self.simulation_manager = SimulationManager('src_client')
        
        # Initialize simulation environment
        self.world_handle = self.simulation_manager.get_world_handle()
        self.scene = Scene(self.simulation_manager)
        self.psm1 = PSM(self.simulation_manager, 'psm1',add_joint_errors=False)
        self.psm2 = PSM(self.simulation_manager, 'psm2',add_joint_errors=False)
        self.psm_list = [self.psm1, self.psm2]
        self.ecm = ECM(self.simulation_manager, 'CameraFrame')
        self.Camera_view_reset()

        self.needle = NeedleInitialization(self.simulation_manager) # needle obj
        self.needle_kin = NeedleKinematics_v2() # needle movement and positioning
        self.init_psm1 = np.array([ 0.04629208,0.00752399,-0.08173992,-3.598019,-0.05762508,1.2738742,0.8],dtype=np.float32)
        self.psm_step(self.init_psm1,1)
        self.init_psm2 = np.array([-0.03721037,  0.01213105, -0.08036895, -2.7039163, 0.07693613, 2.0361109, 0.8],dtype=np.float32)
        self.psm_step(self.init_psm2,2)
        add_break(2.0)
        self.timestep = 0
        print("Initialized!!!")
        return

    def reset(self, **kwargs):
        self.task = 1
        self.subtask_completion = 0
        self.psm1.actuators[0].deactuate()
        self.psm2.actuators[0].deactuate()
        """ Reset the state of the environment to an initial state """
        self.needle_randomization()
        ######################
        low_limits = [-0.02, -0.02, -0.01, -np.deg2rad(30), -np.deg2rad(30), -np.deg2rad(30), -0.1]
        high_limits = [0.02, 0.02, 0.01, np.deg2rad(30), np.deg2rad(30), np.deg2rad(30), 0.1]
        random_array_psm1 = np.random.uniform(low=low_limits, high=high_limits)
        random_array_psm2 = np.random.uniform(low=low_limits, high=high_limits)
        self.psm1_goal = self.init_psm1 
        self.psm2_goal = self.init_psm2 
        self.psm_goal_list = [self.psm1_goal,self.psm2_goal]
        self.psm_step(self.psm1_goal,1)
        self.psm1.set_jaw_angle(0.8)
        time.sleep(0.5)
        self.psm_step(self.psm2_goal,2)
        self.psm2.set_jaw_angle(0.8)
        time.sleep(0.5)
        self.world_handle.reset()
        self.Camera_view_reset()
        add_break(0.5)

        self.needle_obs = self.needle_goal_evaluator(0.010,2)
        self.init_obs_array = np.concatenate((self.init_psm2,self.needle_obs,self.needle_obs-self.init_psm2),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":self.init_psm2,"desired_goal":self.needle_obs}

        self.obs = self.normalize_observation(self.init_obs_dict)
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        self.task_step = 0
        return self.obs, self.info
    
    def step(self, action):
        self.timestep += 1
        self.task_step += 1
        step_size = self.step_size[7*(self.task-1):7*self.task]
        action_step = action*step_size
 
        if (self.subtask_completion):
            action_step*=0

        self.psm_controller = 2 if self.task<4 else 1
        current = self.psm_goal_list[self.psm_controller-1]
        self.psm_goal_list[self.psm_controller-1] = current+action_step

        self.world_handle.update()
        self.psm_step(self.psm_goal_list[self.psm_controller-1],self.psm_controller)
        self.psm_list[self.psm_controller-1].set_jaw_angle(self.psm_goal_list[self.psm_controller-1][-1])
        time.sleep(0.01)
        self.update_observation()
        return self.obs, self.reward, self.terminate, self.truncate, self.info

    
    def render(self, mode='human', close=False):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info=None):
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
                (distances_trans <= self.threshold[2*(self.task-1)]) & (distances_angle <= self.threshold[2*self.task-1]),
                1,  
                0  
            )
        return rewards

    def criteria(self):
        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        distances_trans = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
        if (distances_trans<= self.threshold[2*(self.task-1)]) and (distances_angle <= self.threshold[2*self.task-1]):
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

    def task_update(self,task_idx):
        if (self.task!=task_idx):
            self.task = task_idx
            self.subtask_completion = 0
            self.task_step = 0
            if (task_idx == 1):
                self.psm2.set_jaw_angle(0.8)
                self.psm_goal_list[1][-1] = 0.8
            if (task_idx == 4):
                self.psm1.set_jaw_angle(0.8)
                self.psm_goal_list[0][-1] = 0.8 


    def update_observation(self):
        """ Update the observation of the environment

        Parameters
        - action: an action provided by the environment
        """
        self.terminate = False
        self.truncate = False

        if self.criteria() and self.task_step>5:
            self.subtask_completion = 1
            self.task_step = 0
            if self.task == 1:
                self.psm2.set_jaw_angle(0.0)
                self.psm_goal_list[1][-1] = 0.0
                time.sleep(2.0)
                self.psm2.actuators[0].actuate("Needle") # To be activated
                self.needle.needle.set_force([0.0,0.0,0.0])
                self.needle.needle.set_torque([0.0,0.0,0.0])
                print("Grasping complete")

            elif self.task == 2:
                ## Calibrate the obs:
                psm2_update_pos = self.Frame2Vec(convert_mat_to_frame(self.psm2.measured_cp()))
                self.psm_goal_list[1][:] = np.append(psm2_update_pos,0.0)
                print("Placing complete")

            elif self.task == 3:
                print("Inserting complete")
                time.sleep(2.0)   

            elif self.task == 4:
                self.psm1.set_jaw_angle(0.0)
                self.psm_goal_list[0][-1] = 0.0
                time.sleep(0.5)
                self.psm1.actuators[0].actuate("Needle") # To be activated
                self.needle.needle.set_force([0.0,0.0,0.0])
                self.needle.needle.set_torque([0.0,0.0,0.0])
                time.sleep(1.0)
                self.psm2.set_jaw_angle(0.5)
                self.psm_goal_list[1][-1] = 0.5
                print("Regrasping complete")

            elif self.task == 5:
                print("Pullout complete")
                self.terminate = True
                self.info = {"is_success":True}   


        if self.timestep == self.max_timestep:
            print("Maximum step reaches")

        if self.task == 1 and self.task_step == 1:
            # self.needle_obs = self.needle_goal_evaluator(lift_height=0.010,psm_idx=2) # To be activated
            self.needle_obs = self.needle_goal_evaluator(lift_height=0.008,psm_idx=2)
            self.goal_obs = self.needle_obs
        elif self.task == 2 and self.task_step == 1:
            self.entry_obs = self.entry_goal_evaluator()
            self.goal_obs = self.entry_obs
        elif self.task == 3 and self.task_step == 1:
            self.exit_obs = self.insert_goal_evaluator(110,[0.001,0,0])
            self.goal_obs = self.exit_obs
        elif self.task == 4 and self.task_step == 1:
            self.needle_obs = self.needle_goal_evaluator(lift_height=0.007, psm_idx=1,deg_angle=105)
            self.goal_obs = self.needle_obs
        elif self.task == 5 and self.task_step == 1:
            self.pullout_obs = self.pullout_goal_evaluator()
            self.goal_obs = self.pullout_obs
        
        goal_obs = self.goal_obs    
        current = np.array(self.psm_goal_list[self.psm_controller-1],dtype=np.float32)
        goal_obs = np.array(goal_obs,dtype=np.float32)

        obs_array = np.concatenate((current,goal_obs,goal_obs-current), dtype=np.float32)
        obs_dict = {"observation":obs_array,"achieved_goal":current,"desired_goal":goal_obs}
        self.obs = self.normalize_observation(obs_dict) 

        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        self.reward = self.compute_reward(achieved_goal ,desired_goal) # Already normalized input
        self.reward = float(self.reward)
        self.info = {"is_success":False}

         
    def psm_step(self,obs,psm_idx = 2):
        """
        Given obs = np.array([x,y,z,roll,pitch,yaw,jaw]) 
        move the psm2 to the positions.
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
        origin_p = Vector( -0.0207937, 0.0562045, 0.0711726)
        origin_rz = 0.0


        random_x = np.random.uniform(-0.004, 0.005)
        random_y = np.random.uniform(-0.04, 0.02)
        random_rz = np.random.uniform(-np.pi/4,np.pi/4)

        origin_p[0] += random_x
        origin_p[1] += random_y
        origin_rz += random_rz

        new_rot = Rotation(np.cos(origin_rz),-np.sin(origin_rz),0,
                            np.sin(origin_rz),np.cos(origin_rz),0,
                            0.0,0.0,1.0)
        self.needle_pos_new = Frame(new_rot,origin_p)
        self.needle.needle.set_pose(self.needle_pos_new)
    
    def Frame2Vec(self,goal_frame):
        X_goal = goal_frame.p.x()
        Y_goal = goal_frame.p.y()
        Z_goal = goal_frame.p.z()
        rot_goal = goal_frame.M
        roll_goal,pitch_goal,yaw_goal  = rot_goal.GetRPY()
        if (roll_goal <= np.deg2rad(-360)):
            roll_goal += 2*np.pi
        elif (roll_goal > np.deg2rad(0)):
            roll_goal -= 2*np.pi
        array_goal = np.array([X_goal,Y_goal,Z_goal,roll_goal,pitch_goal,yaw_goal],dtype=np.float32)
        return array_goal
            
    def entry_goal_evaluator(self,idx = 2):
        self.entry_w = self.scene.entry1_measured_cp()
        entry_pos = self.psm_list[idx-1].get_T_w_b()*self.entry_w
        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        T_tip_base = self.needle_kin.get_tip_pose()
        T_gripper_base = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        T_gripper_tip = T_tip_base.Inverse()*T_gripper_base

        T_insert = entry_pos
        T_insert.M *= rotation_entry
        T_insert = T_insert*T_gripper_tip
        array_insert = self.Frame2Vec(T_insert)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def insert_goal_evaluator(self,deg=120,dev=[0,0,0],idx=2):
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
        array_insert = self.Frame2Vec(gripper_in_base)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def pullout_goal_evaluator(self,deg=110,dev=[0,0,0],idx=1):
        exit_in_world = self.scene.exit1_measured_cp()
        rotation_decrease_y = Rotation.RotY(-np.deg2rad(50))
        new_rotation = exit_in_world.M * rotation_decrease_y
        pullout_in_world = pullout_in_world = Frame(new_rotation, Vector(exit_in_world.p[0] + 0.03, exit_in_world.p[1], exit_in_world.p[2] + 0.03))
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*pullout_in_world

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
        array_pullout = self.Frame2Vec(gripper_in_base)
        array_pullout = np.append(array_pullout,0.0)
        return array_pullout
