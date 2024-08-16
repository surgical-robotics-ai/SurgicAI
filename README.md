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

* Install [Stable Baseline3](https://github.com/DLR-RM/stable-baselines3) (SB3) and [d3rlpy](https://github.com/takuseno/d3rlpy): SB3 and d3rlpy are open-sourced Python libraries providing implementations of state-of-the-art RL algorithms. In this project, they are used to interaction with Gymnasium environment and offering interface for training, evaluating, and testing RL models.
```
pip install stable-baselines3 d3rlpy
```

## RL Training
This section introduce the basic procedure for model training with defined Gymnasium environment.

### Run the SRC Environment
Make sure ROS and SRC is running before moving forward to the following steps. You can simply run the following command or refer to this [link](https://github.com/surgical-robotics-ai/surgical_robotics_challenge) for details.

```
roscore
```
```
~/ambf/bin/lin-x86_64/ambf_simulator --launch_file ~/ambf/surgical_robotics_challenge/launch.yaml -l 0,1,3,4,13,14 -p 200 -t 1 --override_max_comm_freq 120 --override_min_comm_freq 120
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
from stable_baselines3.common.noise import NormalActionNoise
import time
gym.envs.register(id="TD3_HER_BC", entry_point=SRC_approach)
env = gym.make("TD3_HER_BC", render_mode="human",reward_type = "sparse")
```

### Initialize and Train the Model
Here is an example of model with TD3+HER+BC. While you may check [Approach_training_HER.ipynb](./Approach_training_HER.ipynb) for more details.

```python
model = TD3_BC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=DemoHerReplayBuffer,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
    replay_buffer_kwargs=dict(
        demo_transitions=episode_transitions, 
        goal_selection_strategy=goal_selection_strategy,
    ),
    episode_transitions=episode_transitions,
)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./First_version/Model_temp', name_prefix='SRC')
model.learn(total_timesteps=int(1000000), progress_bar=True,callback=checkpoint_callback,)
model.save("SRC")
```

### Load the Model
```python
model_path = "./Model/SRC_10000_steps.zip"
model = TD3_BC.load(model_path,env=env)
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

### Low-level Training
The command lines above shows you a brief pipeline of how the pipeline works. In order to train a model specifically for a low level policy, you can directly run with the command below:
```python
python3 RL_training_online.py --algorithm "$algorithm" --task_name "$task" --reward_type "$REWARD_TYPE" --total_timesteps "$TOTAL_TIMESTEPS" --save_freq "$SAVE_FREQ" --seed "$SEED" --trans_error "$TRANS_ERROR" --angle_error "$ANGLE_ERROR"
``` 

### Model Evaluation
The command evaluates the success rate, trajectory length, and time steps across five policies with different random seeds.
```python
python3 Model_evaluation.py --algorithm "$algorithm" --task_name "$task" --reward_type "$REWARD_TYPE" --trans_error "$TRANS_ERROR" --angle_error "$ANGLE_ERROR" --eval_seed "$EVAL_SEED"
```

### High-level Training
See in [High_level_HLP.ipynb](./High_level_HLP.ipynb) for more details.

The following video demonstrates the complete suturing procedure by our training policy.

[demo](https://github.com/surgical-robotics-ai/SurgicAI/assets/147576462/1927a1cf-096f-444d-a878-6c0f96b152d4)

Here's some progress demonstrating our pipeline's transition to the latest SRC, focusing on the low-level task: 'Place'.

[New_SRC_demo](https://github.com/user-attachments/assets/faf0d821-2b6c-4524-be26-565dc2f4a600)

If you find our work userful, please cite it as:
```bibtex
@misc{wu2024surgicaifinegrainedplatformdata,
      title={SurgicAI: A Fine-grained Platform for Data Collection and Benchmarking in Surgical Policy Learning}, 
      author={Jin Wu and Haoying Zhou and Peter Kazanzides and Adnan Munawar and Anqi Liu},
      year={2024},
      eprint={2406.13865},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2406.13865}, 
}


