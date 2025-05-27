import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])

obs, info = env.reset()
print("Observation shape:", obs.shape)
print("Dtype:", obs.dtype)
env.close()

