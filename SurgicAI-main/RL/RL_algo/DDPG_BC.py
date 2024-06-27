from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import torch as th
import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG_BC")


class DDPG_BC(TD3):

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        episode_transitions = None,
        BC_coeff = 0.1,
        demo_ratio = 0.5,
        Q_filter = False,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if episode_transitions is not None:
            self.demo_data = self.preprocess_demo_data(episode_transitions)
            self.BC_coeff = BC_coeff 
            self.demo_ratio = demo_ratio
            self.Q_filter = Q_filter

    def preprocess_demo_data(self, episode_transitions):
        demo_data = {
            "actions": th.tensor([trans["action"] for trans in episode_transitions], dtype=th.float32, device=self.device),
            "observations": {key: [] for key in episode_transitions[0]["obs"].keys()}
        }
        
        for key in demo_data["observations"].keys():
            demo_data["observations"][key] = th.tensor([trans["obs"][key] for trans in episode_transitions], dtype=th.float32, device=self.device)

        return demo_data
    
    def q_filter(self,critic, obs, actor_actions, demo_actions):
        q_actor = critic.q1_forward(obs, actor_actions)
        q_demo = critic.q1_forward(obs, demo_actions)
        return q_demo > q_actor
    
    def learn(
        self: SelfDDPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDPG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            ##### Begin editting
            # success_rate_approx = sum(self.ep_success_buffer)/(max(len(self.ep_success_buffer),30))
            # demo_ratio_dynamic = -0.9*self.demo_ratio*success_rate_approx+self.demo_ratio
            # BC_coeff_dynamic = -0.09*self.BC_coeff*success_rate_approx+self.BC_coeff

            demo_batch_size = int(self.demo_ratio*batch_size)
            demo_indices = th.randint(0, self.demo_data['actions'].size(0), (demo_batch_size,))
            demo_actions = self.demo_data['actions'][demo_indices]
            demo_obs = {key: self.demo_data['observations'][key][demo_indices] for key in self.demo_data['observations'].keys()}
            ##### End editting

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss

                ##### Begin editting
                predicted_actions = self.actor(demo_obs)

                if not self.Q_filter:
                    bc_loss = F.mse_loss(predicted_actions, demo_actions)           
                    rl_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                    actor_loss = self.BC_coeff*bc_loss+(1-self.BC_coeff)*rl_loss
                
                else:
                    q_filter_mask = self.q_filter(self.critic, demo_obs, predicted_actions, demo_actions).squeeze()
                    if q_filter_mask.any():
                        bc_loss = F.mse_loss(predicted_actions[q_filter_mask], demo_actions[q_filter_mask])
                    else:
                        bc_loss = th.tensor(0.0) 
                    rl_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                    actor_loss = self.BC_coeff*bc_loss+(1-self.BC_coeff)*rl_loss
                ##### End editting

                # actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))