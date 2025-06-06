# Run this in Python
import gymnasium as gym
env = gym.make("FetchReach-v3")
obs, _ = env.reset()
print("Observation keys:", obs.keys())

