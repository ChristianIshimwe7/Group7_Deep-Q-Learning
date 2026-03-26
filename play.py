"""
play.py
=======
Load a trained DQN model and play ALE/Breakout-v5.

Examples
--------
python play.py
python play.py --model models/M1_E01_baseline/dqn_model
python play.py --episodes 10
python play.py --no-render --episodes 20
"""

from __future__ import annotations

import argparse
import os
import time

import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


gym.register_envs(ale_py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play ALE/Breakout-v5 with a trained DQN")
    parser.add_argument("--model", type=str, default="dqn_model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.02)
    return parser.parse_args()


def make_env_fn(render_mode: str | None = None):
    def _init():
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        env = AtariWrapper(env)
        return env

    return _init


def build_env(render_mode: str | None = None, n_stack: int = 4, transpose: bool = True):
    env = DummyVecEnv([make_env_fn(render_mode=render_mode)])
    env = VecFrameStack(env, n_stack=n_stack)
    if transpose:
        env = VecTransposeImage(env)
    return env


def load_model(path: str, env) -> DQN:
    # Handle the cases where the user provides an experiment name or a full path
    candidates = [
        path,
        f"{path}.zip",
        f"best/{path}/best_model",
        f"models/{path}/dqn_model",
        "dqn_model.zip",
        "dqn_model",
    ]

    for candidate in candidates:
        if candidate.endswith(".zip") and not os.path.exists(candidate):
            continue
        if os.path.isdir(candidate) and not os.path.exists(os.path.join(candidate, "data")):
            # If it's a directory but not a valid SB3 directory, skip
            continue
            
        try:
            model = DQN.load(candidate, env=env)
            print(f"Loaded model from: {candidate}")
            return model
        except Exception:
            continue

    raise FileNotFoundError(f"Model not found for '{path}'. Checked candidates: {candidates}")


def play(model_path: str, num_episodes: int, render: bool, delay: float) -> None:
    render_mode = "human" if render else None
    
    # Try dummy load to check policy type
    try:
        temp_env = build_env(render_mode=None, transpose=False)
        temp_model = load_model(model_path, temp_env)
        is_cnn = "CnnPolicy" in str(type(temp_model.policy))
    except Exception:
        is_cnn = True # Default to CNN wrappers

    env = build_env(render_mode=render_mode, transpose=is_cnn)
    model = load_model(model_path, env)

    print("=" * 60)
    print("Environment: ALE/Breakout-v5")
    print(f"Policy: {type(model.policy).__name__} (Greedy Q)")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Greedy policy for evaluation.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)

            total_reward += float(reward[0])
            steps += 1

            if render:
                time.sleep(delay)

            if done[0]:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep:>2}/{num_episodes}: steps={steps:>6,} reward={total_reward:>7.2f}")

    env.close()

    print("\nRESULTS")
    print("-" * 60)
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward : {np.std(episode_rewards):.2f}")
    print(f"Max reward : {np.max(episode_rewards):.2f}")
    print(f"Min reward : {np.min(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} steps")
    print("-" * 60)


if __name__ == "__main__":
    args = parse_args()
    play(args.model, args.episodes, not args.no_render, args.delay)
