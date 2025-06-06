import gymnasium as gym
import gymnasium_robotics
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

# Register robotics envs
gym.register_envs(gymnasium_robotics)

# Set env ID and register it for Ray
ENV_ID = "FetchReach-v3"
register_env(ENV_ID, lambda config: gym.make(ENV_ID))

# Create PPO config
config = (
    PPOConfig()
    .environment(env=ENV_ID)
    .framework("torch")  # ✅ PyTorch backend
    .env_runners(num_envs_per_env_runner=1, num_env_runners=2)  # ✅ Updated API
    .training(train_batch_size=4000, model={"fcnet_hiddens": [256, 256]})
)

# Launch training
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"training_iteration": 200},
        name="ppo_fetchreach"
    )
)

if __name__ == "__main__":
    tuner.fit()
