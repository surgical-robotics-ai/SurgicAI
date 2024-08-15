from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy

SelfTD3 = TypeVar("SelfTD3", bound="BC")

class BC(OffPolicyAlgorithm):
    
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        batch_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        episode_transitions = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Box,),
        )

        if _init_setup_model:
            self._setup_model()

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if episode_transitions is not None:
            self.demo_data = self.preprocess_demo_data(episode_transitions)

    def preprocess_demo_data(self, episode_transitions):
        demo_data = {
            "actions": th.tensor([trans["action"] for trans in episode_transitions], dtype=th.float32, device=self.device),
            "observations": {key: [] for key in episode_transitions[0]["obs"].keys()}
        }
        
        for key in demo_data["observations"].keys():
            demo_data["observations"][key] = th.tensor([trans["obs"][key] for trans in episode_transitions], dtype=th.float32, device=self.device)

        return demo_data
    
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer])

        actor_losses = []
        for _ in range(gradient_steps):
            self._n_updates += 1

            demo_indices = th.randint(0, self.demo_data['actions'].size(0), (batch_size,))
            demo_actions = self.demo_data['actions'][demo_indices]
            demo_obs = {key: self.demo_data['observations'][key][demo_indices] for key in self.demo_data['observations'].keys()}

            predicted_actions = self.actor(demo_obs)

            # Calculate BC loss
            actor_loss = F.mse_loss(predicted_actions, demo_actions)
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "actor_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer"]
        return state_dicts, []
