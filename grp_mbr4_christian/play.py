"""
play.py  —  SHARED (all members use the same file)
===================================================
Load a trained DQN model and play ALE/Breakout-v5.
Uses Greedy Q-Policy (deterministic=True).

Usage
-----
    python play.py                          # GUI, 5 episodes, dqn_model.zip
    python play.py --model best/M1_E06_batch_64/best_model
    python play.py --episodes 10
    python play.py --no-render --episodes 20
"""

import argparse
import time
import numpy as np
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

gym.register_envs(ale_py)


def get_args():
    p = argparse.ArgumentParser(description="Play Breakout with trained DQN")
    p.add_argument("--model",     type=str,   default="dqn_model",
                   help="Model path (without .zip). Default: dqn_model")
    p.add_argument("--episodes",  type=int,   default=5,
                   help="Number of episodes (default: 5)")
    p.add_argument("--no-render", action="store_true",
                   help="Disable GUI (headless/server mode)")
    p.add_argument("--delay",     type=float, default=0.02,
                   help="Seconds between frames (default: 0.02)")
    return p.parse_args()


def make_env_fn(render_mode=None):
    def _init():
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        env = AtariWrapper(env)
        return env
    return _init

def build_env(render_mode=None, n_stack=4):
    return VecFrameStack(DummyVecEnv([make_env_fn(render_mode)]), n_stack=n_stack)


def load_model(path, env):
    """Try multiple paths so the script always finds the model."""
    candidates = [
        path,
        f"best/{path}/best_model",
        f"models/{path}/dqn_model",
        "dqn_model",
    ]
    for c in candidates:
        try:
            m = DQN.load(c, env=env)
            print(f"\n  Model loaded from: {c}.zip")
            return m
        except Exception:
            pass
    raise FileNotFoundError(
        "No model found. Run train.py first to create dqn_model.zip"
    )


def play(model_path, num_episodes, render, delay):
    render_mode = "human" if render else None
    env         = build_env(render_mode=render_mode)
    model       = load_model(model_path, env)

    print(f"\n{'─'*60}")
    print(f"  Environment  :  ALE/Breakout-v5")
    print(f"  Policy       :  {type(model.policy).__name__}")
    print(f"  Q-Policy     :  Greedy  (deterministic=True)")
    print(f"  Episodes     :  {num_episodes}")
    print(f"  Rendering    :  {render}")
    print(f"{'─'*60}\n")

    ep_rewards = []
    ep_lengths = []

    for ep in range(1, num_episodes + 1):
        obs       = env.reset()
        done      = False
        total_rew = 0.0
        steps     = 0

        while not done:
            # ── Greedy Q-Policy: always pick highest Q-value action ──
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_rew += float(reward[0])
            steps     += 1
            if render:
                time.sleep(delay)
            if done[0]:
                break

        ep_rewards.append(total_rew)
        ep_lengths.append(steps)
        print(f"  Episode {ep:>3}/{num_episodes}  |  "
              f"steps={steps:>6,}  |  reward={total_rew:>7.2f}")

    env.close()

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  ({num_episodes} episodes)")
    print(f"{'─'*60}")
    print(f"  Mean reward  :  {np.mean(ep_rewards):>8.2f}")
    print(f"  Std  reward  :  {np.std(ep_rewards):>8.2f}")
    print(f"  Max  reward  :  {np.max(ep_rewards):>8.2f}")
    print(f"  Min  reward  :  {np.min(ep_rewards):>8.2f}")
    print(f"  Mean length  :  {np.mean(ep_lengths):>8.1f} steps")
    print(f"{'='*60}\n")

    return ep_rewards


if __name__ == "__main__":
    args = get_args()
    play(
        model_path   = args.model,
        num_episodes = args.episodes,
        render       = not args.no_render,
        delay        = args.delay,
    )
