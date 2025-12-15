import gymnasium as gym
import torch as th
from stable_baselines3 import PPO

env = gym.make("Pendulum-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.save("sac_pendulum_uncompiled")
del model # remove to demonstrate saving and loading

# Method `load` recreates policy in-place, rather than updating weights of current policy
# Below code demonstrates this
model = PPO("MlpPolicy", env, verbose=1)
model.policy = th.compile(model.policy)  # Compile the model
model = PPO.load("sac_pendulum_uncompiled")
