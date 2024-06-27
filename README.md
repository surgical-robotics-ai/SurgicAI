# A Fine-grained Platform for Data Collection and Benchmarking in Surgical Policy Learning

## Prerequiste
This section introduces the necessary configuration you need.
### System Requirement
![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange?style=flat-square&logo=ubuntu) ![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-blue?style=flat-square&logo=github) ![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.2.1-green?style=flat-square&logo=python) ![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-blue?style=flat-square&logo=ros) ![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python) ![Torch](https://img.shields.io/badge/Torch-2.1.0-red?style=flat-square&logo=pytorch) ![ambf](https://img.shields.io/badge/ambf-2.0-yellow?style=flat-square&logo=github)

### Installation
* Install the [surgical robotics challenge environment](https://github.com/surgical-robotics-ai/surgical_robotics_challenge) as well as the AMBF and ROS prerequisites in the link. It provides simulation environment for suturing phantom combined with da Vinci surgical system.
```
git clone https://github.com/surgical-robotics-ai/surgical_robotics_challenge
```
* Install Gymnasium: [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) is a branch and updated version of OpenAI Gym. It provides standard API for the communication between the simulated environment and learning algorithms.
```
pip install gymnasium
```

* Configure the [Pytorch](https://pytorch.org/) and [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (if equipped with NVIDIA card) based on your hardware.

* Install [Stable Baseline3](https://github.com/DLR-RM/stable-baselines3) (SB3): SB3 is an open-source Python library providing implementations of state-of-the-art RL algorithms. In this project, it is used to interaction with Gymnasium environment and offering interface for training, evaluating, and testing RL models.
```
pip install stable-baselines3
```

### Installation Verification
* Try to run the code below. This script will create a embedded RL environment from Gymnasium and train it using the PPO algorithm from Stable Baselines3. If everything is set up correctly, the script should run without any errors.
``` python
import gymnasium
import stable_baselines3

env = gymnasium.make('CartPole-v1')
model = stable_baselines3.PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```

## RL Training
This section introduce the basic procedure for model training with defined Gymnasium environment.

### Run the SRC Environment
Make sure ROS and SRC is running before moving forward to the following steps. You can simply run the following command or refer to this [link](https://github.com/surgical-robotics-ai/surgical_robotics_challenge) for details.

```
roscore
```
```
~/ambf/bin/lin-x86_64/ambf_simulator --launch_file ~/ambf/surgical_robotics_challenge/launch.yaml -l 0,1,3,4,13,14 -p 200 -t 1 --override_max_comm_freq 120
```

### Register the Gymnasium Environment
```python
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from Approach_env import SRC_approach
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from RL_algo.td3_BC import TD3_BC
from RL_algo.DemoHerReplayBuffer import DemoHerReplayBuffer
from stable_baselines3.common.utils import set_random_seed
gym.envs.register(id="TD3_HER_BC", entry_point=SRC_approach)
env = gym.make("TD3_HER_BC", render_mode="human",reward_type = "sparse")
```

### Initialize and Train the Model
Here is an example of model with Proximal Policy Optimization (PPO) algorithm.
```python
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./First_version/",)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./First_version/Model_temp', name_prefix='SRC')
model.learn(total_timesteps=int(1000000), progress_bar=True,callback=checkpoint_callback,)
model.save("SRC")
```

### Load the Model
```python
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./First_version/",)
model_path = "./Model/SRC_10000_steps.zip"
model = PPO.load(model_path)
model.set_env(env=env)
```

### Test the Model Prediction
```python
obs,info = env.reset()
print(obs)
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated,truncated, info = env.step(action)
    print(info)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
``` 
The following video demonstrates the complete suturing procedure by our training policy.

[demo](https://github.com/surgical-robotics-ai/SurgicAI/assets/147576462/1927a1cf-096f-444d-a878-6c0f96b152d4)

