import numpy as np
from Offline_RL_algo.d3rlpy.algos import AWACConfig, BCQConfig, CalQLConfig, CQLConfig, IQLConfig
from Offline_RL_algo.d3rlpy.models import VectorEncoderFactory
import os
import torch

# Constants
DEFAULT_NET_ARCH = [256, 256, 256]
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_USE_GPU = torch.cuda.is_available()

def get_algorithm_config(algorithm_name: str, env, task_name: str, reward_type: str, seed: int, use_gpu: bool):
    """
    Get the configuration for the specified offline RL algorithm.
    :param algorithm_name: Name of the algorithm (CQL, CalQL, IQL, BCQ, AWAC)
    :param env: The environment
    :param task_name: Name of the task
    :param reward_type: Type of reward (dense or sparse)
    :param seed: Random seed
    :param use_gpu: Whether to use GPU
    :return: Instantiated algorithm
    """
    base_path = f"/home/robo/trajectory_data/RL_new/{task_name}/{algorithm_name}/{reward_type}/seed_{seed}"
    os.makedirs(base_path, exist_ok=True)

    # Define custom encoder factory for actor
    actor_encoder = VectorEncoderFactory(
        hidden_units=DEFAULT_NET_ARCH,
        activation='relu',
        use_batch_norm=False,
        dropout_rate=None,
        last_activation='tanh'
    )

    # Define custom encoder factory for critic
    critic_encoder = VectorEncoderFactory(
        hidden_units=DEFAULT_NET_ARCH,
        activation='relu',
        use_batch_norm=False,
        dropout_rate=None,
    )

    base_params = {
        "actor_encoder_factory": actor_encoder,
        "critic_encoder_factory": critic_encoder,
        "actor_learning_rate": DEFAULT_LEARNING_RATE,
        "critic_learning_rate": DEFAULT_LEARNING_RATE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "gamma": 0.99,
    }

    if algorithm_name == "CQL":
        cql_config = CQLConfig(
            **base_params,
            alpha_learning_rate=DEFAULT_LEARNING_RATE,
            n_action_samples=100,
            tau=0.005,
            n_critics=2,
        )
        return cql_config.create(DEFAULT_USE_GPU)
    elif algorithm_name == "CalQL":
        calql_config = CalQLConfig(
            **base_params,
            alpha_learning_rate=DEFAULT_LEARNING_RATE,
            n_action_samples=100,
            tau=0.005,
            n_critics=2,
            initial_alpha=1.0,
        )
        return calql_config.create(DEFAULT_USE_GPU)
    elif algorithm_name == "IQL":
        iql_config = IQLConfig(
            **base_params,
            value_encoder_factory=critic_encoder,
            tau=0.005,
            n_critics=2,
        )
        return iql_config.create(DEFAULT_USE_GPU)
    elif algorithm_name == "BCQ":
        bcq_config = BCQConfig(
            **base_params,
            imitator_learning_rate=DEFAULT_LEARNING_RATE,
            tau=0.005,
            n_action_samples=100,
            lam=0.75,
            action_flexibility=0.05
        )
        return bcq_config.create(DEFAULT_USE_GPU)
    elif algorithm_name == "AWAC":
        awac_config = AWACConfig(
            **base_params,
            tau=0.005,
            lam=1.0,
            n_action_samples=100,
        )
        return awac_config.create(DEFAULT_USE_GPU)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")