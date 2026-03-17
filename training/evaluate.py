import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.maritime_env import MaritimePathEnv


# -----------------------------
# Create vectorized environment
# -----------------------------
env = DummyVecEnv([lambda: MaritimePathEnv()])

# -----------------------------
# Load normalization statistics
# -----------------------------
env = VecNormalize.load("models/env_normalization.pkl", env)

# IMPORTANT: disable training updates
env.training = False
env.norm_reward = False


# -----------------------------
# Load trained model
# -----------------------------
model = PPO.load("models/maritime_autopilot")

print("\nStarting Evaluation\n")

episodes = 3


for ep in range(episodes):

    obs = env.reset()

    total_reward = 0.0
    step_count = 0

    print(f"\nEpisode {ep+1}\n")

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        # unwrap VecEnv outputs
        obs_raw = obs[0]
        reward_val = reward[0]
        done_flag = done[0]

        x = obs_raw[0]
        y = obs_raw[1]
        heading = obs_raw[2]
        distance_to_wp = obs_raw[3]
        obstacle_dist = obs_raw[9]

        print(
            f"Step {step_count:4d} | "
            f"Pos ({x:6.2f},{y:6.2f}) | "
            f"Heading {heading:6.2f} | "
            f"WP dist {distance_to_wp:6.2f} | "
            f"Obs dist {obstacle_dist:6.2f}"
        )

        total_reward += reward_val
        step_count += 1

        if done_flag:
            break

    print("\nEpisode finished")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps: {step_count}\n")


print("Evaluation complete.")