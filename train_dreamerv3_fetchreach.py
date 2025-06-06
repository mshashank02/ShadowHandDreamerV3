"""
DreamerV3 + Gymnasium Robotics (FetchReach-v3) Training Script
Based on Hafner et al., 2023 & 2021 World Model Papers
"""

import gymnasium as gym
import gymnasium_robotics

from ray import tune
from ray.tune import Tuner
from ray.tune.registry import register_env
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config

# ✅ Ensure all robotics environments are registered with Gymnasium
gym.register_envs(gymnasium_robotics)

# ✅ Set environment name (v3 is correct for gymnasium_robotics >= v1.0)
ENV_ID = "FetchReach-v3"

# ✅ Register with Ray explicitly so subprocesses can access it
register_env(ENV_ID, lambda config: gym.make(ENV_ID))

# ✅ Configure DreamerV3
num_gpus = 1  # Set according to your system (can also use 0)
base_config = DreamerV3Config().environment(env=ENV_ID)

# Extract default learning rates
w = base_config.world_model_lr
c = base_config.critic_lr

# Build the full config
config = (
    base_config
    .resources(num_cpus_for_main_process=8 * num_gpus)
    .learners(num_learners=num_gpus, num_gpus_per_learner=1)
    .env_runners(num_envs_per_env_runner=8 * num_gpus, remote_worker_envs=True)
    .reporting(
        metrics_num_episodes_for_smoothing=num_gpus,
        report_images_and_videos=False,
        report_dream_data=False,
        report_individual_batch_item_stats=False,
    )
    .training(
        model_size="S",  # Use "S" for stability; change to "XL" for final runs
        training_ratio=64,
        batch_size_B=16 * num_gpus,
        world_model_lr=[[0, 0.4 * w], [50000, 0.4 * w], [100000, 3 * w]],
        critic_lr=[[0, 0.4 * c], [50000, 0.4 * c], [100000, 3 * c]],
        actor_lr=[[0, 0.4 * c], [50000, 0.4 * c], [100000, 3 * c]],
    )
)

# ✅ Launch training with Tuner
tuner = Tuner(
    "DreamerV3",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        name="dreamerv3_fetchreach",
        stop={"training_iteration": 200},  # Adjust as needed
        verbose=1,
    ),
)

# ✅ Entry point
if __name__ == "__main__":
    print("🚀 Starting DreamerV3 training on FetchReach-v3...")
    tuner.fit()
