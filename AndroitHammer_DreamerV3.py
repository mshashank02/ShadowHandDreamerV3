"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

try:
    import gymnasium_robotics  # noqa
except (ImportError, ModuleNotFoundError):
    print("You have to `pip install gymnasium_robotics` in order to run this example!")

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray import tune

class BoxObsFloat32Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.array(space.low, dtype=np.float32),
            high=np.array(space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32)
# ------------------------------------------------
# ✅ Register the wrapped environment for RLlib
# ------------------------------------------------
tune.register_env("flappy-bird", lambda ctx: BoxObsFloat32Wrapper(gym.make("AdroitHandHammer-v1")))

# ------------------------------------------------
# ⚙️ DreamerV3 configuration
# ------------------------------------------------
num_gpus = 4
config = DreamerV3Config()
w = config.world_model_lr
c = config.critic_lr

(
    config.resources(
        num_cpus_for_main_process=8 * num_gpus,
    )
    .learners(
        num_learners=num_gpus,
        num_gpus_per_learner=1,
    )
    .env_runners(
        num_envs_per_env_runner=8 * num_gpus,
        remote_worker_envs=True
    )
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

# ------------------------------------------------
# ✅ Optional Sanity Check for Observation Wrapper
# ------------------------------------------------
if __name__ == "__main__":
    print("🔍 Running sanity check for BoxObsFloat32Wrapper...")
    test_env = BoxObsFloat32Wrapper(gym.make("AdroitHandHammer-v1"))
    test_obs, _ = test_env.reset()
    print(f"Obs: dtype={test_obs.dtype}, shape={test_obs.shape}")
    is_valid = test_env.observation_space.contains(test_obs)
    print(f"Valid in obs space: {is_valid}")
    print("✅ Sanity check complete.")