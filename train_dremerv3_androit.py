"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

import gymnasium as gym
from ray import tune
from ray.tune import Tuner
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config

try:
    import gymnasium_robotics  # Ensure envs are available
except ImportError:
    raise ImportError("Install with: `pip install gymnasium_robotics`")

gym.register_envs(gymnasium_robotics)
# Set environment ID
ENV_ID = "FetchReach-v3"

# Register the environment
tune.register_env(ENV_ID, lambda ctx: gym.make(ENV_ID))

# GPU config
num_gpus = 1  # Set to 1 or 4 based on your system

# Build DreamerV3Config
config = DreamerV3Config().environment(env=ENV_ID)
w = config.world_model_lr
c = config.critic_lr

config = (
    config.resources(num_cpus_for_main_process=8 * num_gpus)
    .learners(num_learners=num_gpus, num_gpus_per_learner=1)
    .env_runners(num_envs_per_env_runner=8 * num_gpus, remote_worker_envs=True)
    .reporting(
        metrics_num_episodes_for_smoothing=num_gpus,
        report_images_and_videos=False,
        report_dream_data=False,
        report_individual_batch_item_stats=False,
    )
    .training(
        model_size="XL",
        training_ratio=64,
        batch_size_B=16 * num_gpus,
        world_model_lr=[[0, 0.4 * w], [50000, 0.4 * w], [100000, 3 * w]],
        critic_lr=[[0, 0.4 * c], [50000, 0.4 * c], [100000, 3 * c]],
        actor_lr=[[0, 0.4 * c], [50000, 0.4 * c], [100000, 3 * c]],
    )
)

# Optional: Reduce model size for debugging
# config = config.training(model_size="S")

# Convert to Tuner and run
tuner = Tuner(
    "DreamerV3",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        name="dreamerv3_adroit",
        stop={"training_iteration": 200},  # Customize as needed
        verbose=1,
    ),
)

if __name__ == "__main__":
    tuner.fit()
