import gymnasium as gym
import gymnasium_robotics
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config

# ✅ Register all envs properly (must happen before Ray runs workers)
gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReach-v3"  # Confirmed from doc

# ✅ Explicitly re-register using a constructor lambda for Ray
register_env(ENV_ID, lambda config: gym.make(ENV_ID))

# ✅ Now use ENV_ID normally in DreamerV3Config
config = DreamerV3Config().environment(env=ENV_ID)
