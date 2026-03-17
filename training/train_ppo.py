"""
train_ppo.py  —  Curriculum Training (Phase 1 + Phase 2 automatic)
===================================================================

Phase 1  —  2 obstacles, 4M steps, train from scratch
Phase 2  —  3 obstacles, 2M steps, fine-tune from Phase 1 best

Usage:
  python training/train_ppo.py
  python training/train_ppo.py --phase 1
  python training/train_ppo.py --phase 2
"""

import sys
import os
import shutil
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)

from env.maritime_env import MaritimePathEnv


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------

def make_env(num_obstacles, rank=0, seed=0):

    def _init():
        env = MaritimePathEnv()
        env.obstacles.set_num_obstacles(num_obstacles)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_envs(num_obstacles, num_envs=4, seed=42):

    return DummyVecEnv([
        make_env(num_obstacles, rank=i, seed=seed)
        for i in range(num_envs)
    ])


def make_eval_env(num_obstacles):

    return DummyVecEnv([
        make_env(num_obstacles, rank=99, seed=999)
    ])


# ------------------------------------------------------------------
# PHASE 1
# ------------------------------------------------------------------

def run_phase1(num_envs=4):

    print("\n" + "=" * 60)
    print("PHASE 1 — 2 obstacles  |  4M steps  |  scratch training")
    print("=" * 60 + "\n")

    os.makedirs("models/best", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_vec_envs(num_obstacles=2, num_envs=num_envs)
    eval_env = make_eval_env(num_obstacles=2)

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // num_envs, 1),
        save_path="./models/",
        name_prefix="phase1_ckpt",
        verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval_phase1/",
        eval_freq=max(100_000 // num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        device="auto",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        ent_coef=0.003,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],
                vf=[256, 256, 128]
            )
        ),
        tensorboard_log="./logs/"
    )

    model.learn(
        total_timesteps=4_000_000,
        callback=CallbackList([checkpoint_cb, eval_cb]),
        progress_bar=True
    )

    model.save("models/maritime_autopilot_phase1")
    model.save("models/maritime_autopilot")

    env.save("models/env_normalization_phase1.pkl")
    env.save("models/env_normalization.pkl")

    print("\nPhase 1 complete.")
    print("Best model → models/best/best_model.zip\n")


# ------------------------------------------------------------------
# PHASE 2
# ------------------------------------------------------------------

def run_phase2(num_envs=4):

    print("\n" + "=" * 60)
    print("PHASE 2 — 3 obstacles  |  2M steps  |  fine-tune from Phase 1")
    print("=" * 60 + "\n")

    os.makedirs("models/best_phase2", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists("models/best/best_model.zip"):
        print("ERROR: Phase 1 best model not found.")
        print("Run Phase 1 first.")
        return

    env = make_vec_envs(num_obstacles=3, num_envs=num_envs)
    eval_env = make_eval_env(num_obstacles=3)

    env = VecNormalize.load("models/env_normalization.pkl", env)
    eval_env = VecNormalize.load("models/env_normalization.pkl", eval_env)

    env.training = True
    env.norm_reward = True

    eval_env.training = False
    eval_env.norm_reward = False

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // num_envs, 1),
        save_path="./models/",
        name_prefix="phase2_ckpt",
        verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_phase2/",
        log_path="./logs/eval_phase2/",
        eval_freq=max(100_000 // num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )

    # FIX: removed .zip extension
    model = PPO.load(
        "models/best/best_model",
        env=env,
        verbose=1,
        device="auto",
        learning_rate=1e-4,
        ent_coef=0.001,
        tensorboard_log="./logs/"
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=CallbackList([checkpoint_cb, eval_cb]),
        progress_bar=True,
        reset_num_timesteps=False
    )

    model.save("models/maritime_autopilot")
    env.save("models/env_normalization.pkl")

    phase2_best = "models/best_phase2/best_model.zip"
    final_best = "models/best/best_model.zip"

    if os.path.exists(phase2_best):

        if os.path.exists(final_best):
            shutil.copy(final_best, "models/best/best_model_phase1.zip")

        shutil.copy(phase2_best, final_best)

        print("\nPhase 2 best promoted → models/best/best_model.zip")

    else:

        print("\nWARNING: Phase2 best not found. Saving final weights.")
        model.save("models/best/best_model")

    print("\nTraining complete.")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=both phases, 1=phase1 only, 2=phase2 only"
    )

    args = parser.parse_args()

    if args.phase == 0:
        run_phase1()
        run_phase2()

    elif args.phase == 1:
        run_phase1()

    elif args.phase == 2:
        run_phase2()


if __name__ == "__main__":
    main()