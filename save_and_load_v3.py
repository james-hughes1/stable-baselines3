# Code snippet from @araffin on issue #2137

import torch as th
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "Pendulum-v1", verbose=1)
model.policy = th.compile(model.policy)
model.save("ppo_pendulum_compiled")
model = PPO.load("ppo_pendulum_compiled")