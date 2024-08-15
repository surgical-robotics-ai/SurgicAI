import numpy as np
import torch as th
from typing import Optional, Dict
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer, DictReplayBufferSamples

class DemoHerReplayBuffer(HerReplayBuffer):
    def __init__(self, *args, demo_transitions=None, demo_sample_ratio=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_transitions = demo_transitions if demo_transitions is not None else []
        self.demo_sample_ratio = demo_sample_ratio
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        if demo_transitions is not None:
            self.demo_data = self.preprocess_demo_data(demo_transitions)
        else:
            self.demo_data = None

    def add(self, obs, action, reward, next_obs, done, infos):
        super().add(obs, action, reward, next_obs, done, infos)
        
    def preprocess_demo_data(self, demo_transitions):
        observations = {'observation': [], 'achieved_goal': [], 'desired_goal': []}
        actions = []
        next_observations = {'observation': [], 'achieved_goal': [], 'desired_goal': []}
        dones = []
        rewards = []

        for transition in demo_transitions:
            for key in observations:
                observations[key].append(transition['obs'][key])
                next_observations[key].append(transition['next_obs'][key])
            actions.append(transition['action'])
            dones.append(transition['done'])
            rewards.append(transition['reward'])

        for key in observations:
            observations[key] = th.tensor(observations[key], dtype=th.float32, device=self.device)
            next_observations[key] = th.tensor(next_observations[key], dtype=th.float32, device=self.device)
        actions = th.tensor(actions, dtype=th.float32, device=self.device)
        dones = th.tensor(dones, dtype=th.float32, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)

        return {
            'observations': observations,
            'actions': actions,
            'next_observations': next_observations,
            'dones': dones,
            'rewards': rewards
        }
    
    def index_nested_dict(self, nested_dict, indices):
        if isinstance(nested_dict, dict):
            return {key: self.index_nested_dict(value, indices) for key, value in nested_dict.items()}
        else:
            return nested_dict[indices]

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # 修改: 添加对 batch_size 为 0 的处理
        if batch_size == 0:
            return self._get_empty_sample()

        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError(
                "Unable to sample before the end of the first episode. We recommend choosing a value "
                "for learning_starts that is greater than the maximum number of timesteps in the environment."
            )
        
        demo_batch_size = int(batch_size * self.demo_sample_ratio)
        env_batch_size = batch_size - demo_batch_size

        # 修改: 确保在使用 demo_samples 之前它已经被定义
        demo_samples = {}
        if self.demo_data and demo_batch_size > 0:
            demo_indices = th.randint(0, self.demo_data['actions'].size(0), (demo_batch_size,))
            demo_samples = {key: self.index_nested_dict(value, demo_indices) for key, value in self.demo_data.items()}

        valid_indices = np.flatnonzero(is_valid)
        sampled_indices = np.random.choice(valid_indices, size=env_batch_size, replace=True)
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

        nb_virtual = int(self.her_ratio * env_batch_size)
        virtual_batch_indices, real_batch_indices = np.split(batch_indices, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)
        virtual_data = self._get_virtual_samples(virtual_batch_indices, virtual_env_indices, env)

        observations = {
            key: th.cat((real_data.observations[key], virtual_data.observations[key], demo_samples.get("observations", {}).get(key, th.empty(0, device=self.device))))
            for key in real_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions, demo_samples.get("actions", th.empty(0, device=self.device))))
        next_observations = {
            key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key], demo_samples.get("next_observations", {}).get(key, th.empty(0, device=self.device))))
            for key in real_data.next_observations.keys()
        }
        dones = th.cat((real_data.dones, virtual_data.dones, demo_samples.get("dones", th.empty(0, device=self.device))))
        rewards = th.cat((real_data.rewards, virtual_data.rewards, demo_samples.get("rewards", th.empty(0, device=self.device))))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def _get_empty_sample(self) -> DictReplayBufferSamples:
        empty_tensor = th.empty(0, device=self.device)
        empty_dict = {key: th.empty((0,) + tuple(shape), device=self.device) for key, shape in self.observation_shape.items()}
        return DictReplayBufferSamples(
            observations=empty_dict,
            actions=empty_tensor,
            next_observations=empty_dict,
            dones=empty_tensor,
            rewards=empty_tensor,
        )