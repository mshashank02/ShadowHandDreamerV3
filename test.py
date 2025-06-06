import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchReach-v3', max_episode_steps=100)
obs, _ = env.reset()
print("Observation keys:", obs.keys())

