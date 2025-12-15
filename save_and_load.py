import gymnasium as gym
import torch as th
from stable_baselines3 import PPO

env = gym.make("Pendulum-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.save("sac_pendulum_uncompiled")
del model # remove to demonstrate saving and loading

# Success
model = PPO.load("sac_pendulum_uncompiled")
del model # remove to demonstrate saving and loading

model = PPO("MlpPolicy", env, verbose=1)
model.policy = th.compile(model.policy)  # Compile the model
model.save("sac_pendulum_compiled")
del model # remove to demonstrate saving and loading

# Fail
model = PPO.load("sac_pendulum_compiled")
del model # remove to demonstrate saving and loading