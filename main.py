import numpy as np
from env.maritime_env import MaritimePathEnv


env = MaritimePathEnv()

obs, _ = env.reset()

print("Initial State:", obs)

for step in range(20):

    # random rudder command
    action = np.array([np.random.uniform(-1, 1)])

    obs, reward, terminated, truncated, _ = env.step(action)

    env.render()

    if terminated or truncated:
        print("Episode finished")
        break