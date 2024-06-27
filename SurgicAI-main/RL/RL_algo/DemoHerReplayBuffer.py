import numpy as np
import torch as th
from typing import Optional
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

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:  # type: ignore[override]
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.
        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError(
                "Unable to sample before the end of the first episode. We recommend choosing a value "
                "for learning_starts that is greater than the maximum number of timesteps in the environment."
            )
        
        #######begin editting
        demo_batch_size = int(batch_size * self.demo_sample_ratio)
        env_batch_size = batch_size - demo_batch_size

        if self.demo_data and demo_batch_size > 0:
            demo_indices = th.randint(0, self.demo_data['actions'].size(0), (demo_batch_size,))
            demo_samples = {key: self.index_nested_dict(value, demo_indices) for key, value in self.demo_data.items()}

        ########End editing

        # Get the indices of valid transitions
        # Example:
        # if is_valid = [[True, False, False], [True, False, True]],
        # is_valid has shape (buffer_size=2, n_envs=3)
        # then valid_indices = [0, 3, 5]
        # they correspond to is_valid[0, 0], is_valid[1, 0] and is_valid[1, 2]
        # or in numpy format ([rows], [columns]): (array([0, 1, 1]), array([0, 0, 2]))
        # Those indices are obtained back using np.unravel_index(valid_indices, is_valid.shape)
        valid_indices = np.flatnonzero(is_valid)
        # Sample valid transitions that will constitute the minibatch of size batch_size
        sampled_indices = np.random.choice(valid_indices, size=env_batch_size, replace=True)
        # Unravel the indexes, i.e. recover the batch and env indices.
        # Example: if sampled_indices = [0, 3, 5], then batch_indices = [0, 1, 1] and env_indices = [0, 0, 2]
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

        # Split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * env_batch_size)
        virtual_batch_indices, real_batch_indices = np.split(batch_indices, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # Get real and virtual data
        real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)
        # Create virtual transitions by sampling new desired goals and computing new rewards
        virtual_data = self._get_virtual_samples(virtual_batch_indices, virtual_env_indices, env)


        #############begin editing

        # # Concatenate real and virtual data
        # observations = {
        #     key: th.cat((real_data.observations[key], virtual_data.observations[key]))
        #     for key in virtual_data.observations.keys()
        # }
        # actions = th.cat((real_data.actions, virtual_data.actions))
        # next_observations = {
        #     key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
        #     for key in virtual_data.next_observations.keys()
        # }
        # dones = th.cat((real_data.dones, virtual_data.dones))
        # rewards = th.cat((real_data.rewards, virtual_data.rewards))

        observations = {
            key: th.cat((real_data.observations[key], virtual_data.observations[key], demo_samples["observations"][key]))
            for key in real_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions, demo_samples["actions"]))
        next_observations = {
            key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key], demo_samples["next_observations"][key]))
            for key in real_data.next_observations.keys()
        }
        dones = th.cat((real_data.dones, virtual_data.dones, demo_samples["dones"]))
        rewards = th.cat((real_data.rewards, virtual_data.rewards, demo_samples["rewards"]))
        ############end editing

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )
    

    # def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:  # type: ignore[override]
    #     """
    #     Sample elements from the replay buffer.

    #     :param batch_size: Number of element to sample
    #     :param env: Associated VecEnv to normalize the observations/rewards when sampling
    #     :return: Samples
    #     """
    #     # When the buffer is full, we rewrite on old episodes. We don't want to
    #     # sample incomplete episode transitions, so we have to eliminate some indexes.
    #     is_valid = self.ep_length > 0
    #     if not np.any(is_valid):
    #         raise RuntimeError(
    #             "Unable to sample before the end of the first episode. We recommend choosing a value "
    #             "for learning_starts that is greater than the maximum number of timesteps in the environment."
    #         )
        
    #     #######begin editting
    #     demo_batch_size = int(batch_size * self.demo_sample_ratio)
    #     env_batch_size = batch_size - demo_batch_size
    #     if self.demo_transitions and demo_batch_size > 0:
    #         demo_indices = np.random.randint(0, len(self.demo_transitions), demo_batch_size)
    #         demo_samples = [self.demo_transitions[i] for i in demo_indices]

    #     demo_samples = {
    #         "observations": {
    #             "observation": th.tensor([sample["obs"]["observation"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #             "achieved_goal": th.tensor([sample["obs"]["achieved_goal"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #             "desired_goal": th.tensor([sample["obs"]["desired_goal"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #         },
    #         "actions": th.tensor([sample["action"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #         "next_observations": {
    #             "observation": th.tensor([sample["next_obs"]["observation"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #             "achieved_goal": th.tensor([sample["next_obs"]["achieved_goal"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #             "desired_goal": th.tensor([sample["next_obs"]["desired_goal"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #         },
    #         "dones": th.tensor([sample["done"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #         "rewards": th.tensor([sample["reward"] for sample in demo_samples], dtype=th.float32, device=self.device),
    #     }
    #     ########End editing

    #     # Get the indices of valid transitions
    #     # Example:
    #     # if is_valid = [[True, False, False], [True, False, True]],
    #     # is_valid has shape (buffer_size=2, n_envs=3)
    #     # then valid_indices = [0, 3, 5]
    #     # they correspond to is_valid[0, 0], is_valid[1, 0] and is_valid[1, 2]
    #     # or in numpy format ([rows], [columns]): (array([0, 1, 1]), array([0, 0, 2]))
    #     # Those indices are obtained back using np.unravel_index(valid_indices, is_valid.shape)
    #     valid_indices = np.flatnonzero(is_valid)
    #     # Sample valid transitions that will constitute the minibatch of size batch_size
    #     sampled_indices = np.random.choice(valid_indices, size=env_batch_size, replace=True)
    #     # Unravel the indexes, i.e. recover the batch and env indices.
    #     # Example: if sampled_indices = [0, 3, 5], then batch_indices = [0, 1, 1] and env_indices = [0, 0, 2]
    #     batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

    #     # Split the indexes between real and virtual transitions.
    #     nb_virtual = int(self.her_ratio * env_batch_size)
    #     virtual_batch_indices, real_batch_indices = np.split(batch_indices, [nb_virtual])
    #     virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

    #     # Get real and virtual data
    #     real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)
    #     # Create virtual transitions by sampling new desired goals and computing new rewards
    #     virtual_data = self._get_virtual_samples(virtual_batch_indices, virtual_env_indices, env)


    #     #############begin editing

    #     # # Concatenate real and virtual data
    #     # observations = {
    #     #     key: th.cat((real_data.observations[key], virtual_data.observations[key]))
    #     #     for key in virtual_data.observations.keys()
    #     # }
    #     # actions = th.cat((real_data.actions, virtual_data.actions))
    #     # next_observations = {
    #     #     key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
    #     #     for key in virtual_data.next_observations.keys()
    #     # }
    #     # dones = th.cat((real_data.dones, virtual_data.dones))
    #     # rewards = th.cat((real_data.rewards, virtual_data.rewards))

    #     observations = {
    #         key: th.cat((real_data.observations[key], virtual_data.observations[key], demo_samples["observations"][key]))
    #         for key in real_data.observations.keys()
    #     }
    #     actions = th.cat((real_data.actions, virtual_data.actions, demo_samples["actions"]))
    #     next_observations = {
    #         key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key], demo_samples["next_observations"][key]))
    #         for key in real_data.next_observations.keys()
    #     }
    #     dones = th.cat((real_data.dones, virtual_data.dones, demo_samples["dones"]))
    #     rewards = th.cat((real_data.rewards, virtual_data.rewards, demo_samples["rewards"]))
    #     ############end editing

    #     return DictReplayBufferSamples(
    #         observations=observations,
    #         actions=actions,
    #         next_observations=next_observations,
    #         dones=dones,
    #         rewards=rewards,
    #     )
    
