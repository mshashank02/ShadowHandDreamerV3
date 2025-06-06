"""
DreamerV3 + CartPole-v1 (Standalone Training Script)

Papers:
[1] Mastering Diverse Domains through World Models - 2023
    https://arxiv.org/pdf/2301.04104v1.pdf
[2] Mastering Atari with Discrete World Models - 2021
    https://arxiv.org/pdf/2010.02193.pdf
"""

from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray import tune
from ray.tune import Tuner
from ray.tune.registry import register_env
import gymnasium as gym

# ✅ Define environment
ENV_ID = "CartPole-v1"

# ✅ Register env for Ray workers (important for subprocesses)
register_env(ENV_ID, lambda cfg: gym.make(ENV_ID))

# ✅ Build DreamerV3 config
config = (
    DreamerV3Config()
    .environment(env=ENV_ID)
    .resources(num_cpus_for_main_process=2)
    .learners(num_learners=1, num_gpus_per_learner=0)  # No GPU needed for CartPole
    .env_runners(num_envs_per_env_runner=2, remote_worker_envs=True)
    .training(
        model_size="XS",
        training_ratio=1024,
        batch_size_B=16,
    )
)

# ✅ Define tuner to run training
tuner = Tuner(
    "DreamerV3",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        name="dreamerv3_cartpole",
        stop={"training_iteration": 100},  # You can increase this later
        verbose=1,
    ),
)

# ✅ Run training
if __name__ == "__main__":
    print("🚀 Starting DreamerV3 training on CartPole-v1...")
    tuner.fit()
