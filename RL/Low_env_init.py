import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
# from EasyEnv import myEasyGym
from Low_level_env_complete import SRC_low_level
import numpy as np
from stable_baselines3.common.env_checker import check_env
from RL_algo.td3_BC import TD3_BC
from stable_baselines3.common.utils import set_random_seed
from gymnasium import spaces
import time
# Create environment

def low_pass_filter(prev_action, new_action, alpha=0.5):
    """
    Apply low pass filter
    alpha: smooth factor
    """
    return alpha * new_action + (1 - alpha) * prev_action

class DummyEnv(gym.Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
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

class low_level_controller():
    def __init__(self,seed=10):
        set_random_seed(seed)


        exp_index = 4
        txtfile1 = '/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2/high_level_step_size.txt'
        step_size= np.loadtxt(txtfile1, dtype=np.float32)
        txtfile2 = '/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2/threshold_high_level_complete.txt'
        threshold = np.loadtxt(txtfile2, dtype=np.float32)

        episode_steps = 800

        gym.envs.register(id="low_level", entry_point=SRC_low_level, max_episode_steps=episode_steps)
        self.env = gym.make("low_level", render_mode="human",reward_type = "sparse",max_episode_step=episode_steps,seed = seed, step_size=step_size,threshold=threshold)

        dummy_env = DummyEnv()
        self.prev_action = None

        self.model_Approach = TD3_BC(
            "MultiInputPolicy",
            dummy_env,
        )

        # Viable param
        self.model_insert = TD3_BC(
            "MultiInputPolicy",
            dummy_env,
        )

        self.model_pass = TD3_BC(
            "MultiInputPolicy",
            dummy_env,
        )

        self.model_regrasp = TD3_BC(
            "MultiInputPolicy",
            dummy_env,
        )

        self.model_handover = TD3_BC(
            "MultiInputPolicy",
            dummy_env,
        )


        model_path_approach = "/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2/Approach/TD3_BC_noise_dense/rl_model_final.zip"
        self.model_approach = TD3_BC.load(model_path_approach,env=dummy_env)#

        model_path_place = "/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2/Place/TD3_BC_noise_sparse/rl_model_final.zip"
        self.model_place = TD3_BC.load(model_path_place,env=dummy_env)

        model_path_insert = "/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2/Insert/TD3_BC_sparse/rl_model_final.zip"
        self.model_insert = TD3_BC.load(model_path_insert,env=dummy_env)

        model_path_regrasp = "/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2/Regrasp/TD3_BC_noise_dense/rl_model_final.zip"
        self.model_regrasp = TD3_BC.load(model_path_regrasp,env=dummy_env)

        model_path_pullout = "/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2/Pullout/TD3_BC_noise_dense/rl_model_final.zip"
        self.model_pullout = TD3_BC.load(model_path_pullout,env=dummy_env)


    def low_level_step(self,obs_low,filter = 0):
        if (self.env.task == 1):
            # print("Grasping model")
            action, _ = self.model_approach.predict(obs_low, deterministic=True)
            action[-1] = 0
        
        elif (self.env.task == 2):
            # print("Inserting model")
            action, _ = self.model_place.predict(obs_low, deterministic=True)
            action[-1] = 0
        
        elif (self.env.task == 3):
            # print("Passing model")
            action, _ = self.model_insert.predict(obs_low, deterministic=True)
            action[-1] = 0      
        
        elif (self.env.task == 4):
            # print("Regrasping model")
            action, _ = self.model_regrasp.predict(obs_low, deterministic=True)
            action[-1] = 0

        elif (self.env.task == 5):
            # print("Handover model")
            action, _ = self.model_pullout.predict(obs_low, deterministic=True)
            action[-1] = 0   

        # print(f"action= {action}")
        if filter:
            if self.prev_action is not None:
                action = low_pass_filter(self.prev_action, action)
            self.prev_action = action


        # if self.env.timestep == 80:
        #     self.env.psm2.set_jaw_angle(0.5)
        #     self.env.psm2_goal[-1] = 0.5
        #     self.subtask_completion = 1
        #     self.env.needle.needle.set_pose(self.env.needle_pos_new)
        #     self.env.threshold[0] = 0.04
        #     self.env.update_observation()
            
        next_obs_low, reward, terminated, truncated, info = self.env.step(action) 
        time.sleep(0.01)
        return next_obs_low, reward, terminated, truncated, info
    
