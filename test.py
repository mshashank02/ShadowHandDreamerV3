import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("AdroitHandHammer-v1")
obs, _ = env.reset()

print("Observation type:", type(obs))  # <class 'dict'>
