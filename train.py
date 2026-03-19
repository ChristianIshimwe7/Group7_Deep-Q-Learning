"""
train.py
========
Train a DQN agent for ALE/Breakout-v5 using Stable Baselines3.

Examples
--------
# Run all 10 experiments for member 1
python train.py --member 1

# Run a single experiment for member 1
python train.py --member 1 --exp 1

# Quick smoke test
python train.py --member 1 --exp 1 --timesteps 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)


gym.register_envs(ale_py)

RESULTS_CSV = "results/experiments.csv"


@dataclass
class ExperimentConfig:
    experiment_name: str
    policy: str = "CnnPolicy"
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    exploration_fraction: float = 0.10


class TrainingLogger(BaseCallback):
    """Track episode rewards and lengths for quick trend inspection."""

    def __init__(self, log_interval: int = 10_000) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self._cur_rew: dict[int, float] = {}
        self._cur_len: dict[int, int] = {}

    def _on_step(self) -> bool:
        for i, (done, rew) in enumerate(zip(self.locals["dones"], self.locals["rewards"])):
            self._cur_rew[i] = self._cur_rew.get(i, 0.0) + float(rew)
            self._cur_len[i] = self._cur_len.get(i, 0) + 1
            if done:
                self.ep_rewards.append(self._cur_rew[i])
                self.ep_lengths.append(self._cur_len[i])
                self._cur_rew[i] = 0.0
                self._cur_len[i] = 0

        if self.n_calls % self.log_interval == 0 and self.ep_rewards:
            last_rewards = self.ep_rewards[-20:]
            last_lengths = self.ep_lengths[-20:]
            print(
                f"[step {self.num_timesteps:>8,}] "
                f"mean_reward(last20)={np.mean(last_rewards):.2f} "
                f"mean_len(last20)={np.mean(last_lengths):.1f} "
                f"episodes={len(self.ep_rewards)}"
            )
        return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_fn(seed: int | None = None, render_mode: str | None = None):
    def _init():
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        env = AtariWrapper(env)
        return env

    return _init


def build_env(
    seed: int | None = None,
    render_mode: str | None = None,
    n_stack: int = 4,
    transpose_image: bool = True,
):
    env = DummyVecEnv([make_env_fn(seed=seed, render_mode=render_mode)])
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecMonitor(env)
    if transpose_image:
        env = VecTransposeImage(env)
    return env


def append_results_csv(row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    file_exists = os.path.exists(RESULTS_CSV)

    fieldnames = [
        "member",
        "experiment_number",
        "experiment_name",
        "tag",
        "policy",
        "learning_rate",
        "gamma",
        "batch_size",
        "exploration_initial_eps",
        "exploration_final_eps",
        "exploration_fraction",
        "timesteps",
        "buffer_size",
        "seed",
        "mean_reward_last20",
        "mean_episode_len_last20",
        "total_episodes",
        "model_path",
        "best_model_path",
        "log_path",
    ]

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(
    *,
    member: int,
    exp_num: int,
    cfg: ExperimentConfig,
    total_timesteps: int,
    seed: int,
    buffer_size: int,
) -> dict[str, Any]:
    tag = f"M{member}_E{exp_num:02d}_{cfg.experiment_name}"
    model_dir = f"models/{tag}"
    best_dir = f"best/{tag}"
    log_dir = f"logs/{tag}"
    tb_dir = f"tensorboard/{tag}"

    for path in [model_dir, best_dir, log_dir, tb_dir]:
        os.makedirs(path, exist_ok=True)

    print("=" * 72)
    print(
        f"Member={member}  Experiment=E{exp_num:02d} ({cfg.experiment_name})  "
        f"Policy={cfg.policy}"
    )
    print(
        f"lr={cfg.learning_rate} gamma={cfg.gamma} batch={cfg.batch_size} "
        f"eps={cfg.exploration_initial_eps}->{cfg.exploration_final_eps} "
        f"frac={cfg.exploration_fraction} timesteps={total_timesteps} seed={seed} "
        f"buffer={buffer_size}"
    )
    print("=" * 72)

    transpose_image = cfg.policy == "CnnPolicy"
    train_env = build_env(seed=seed, transpose_image=transpose_image)
    eval_env = build_env(seed=seed + 123, transpose_image=transpose_image)

    model = DQN(
        policy=cfg.policy,
        env=train_env,
        learning_rate=cfg.learning_rate,
        buffer_size=buffer_size,
        learning_starts=10_000,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=cfg.exploration_fraction,
        exploration_initial_eps=cfg.exploration_initial_eps,
        exploration_final_eps=cfg.exploration_final_eps,
        optimize_memory_usage=False,
        tensorboard_log=tb_dir,
        seed=seed,
        verbose=0,
    )

    logger = TrainingLogger(log_interval=10_000)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=log_dir,
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger, eval_cb],
        progress_bar=True,
    )

    model.save(f"{model_dir}/dqn_model")

    # Keep assignment-required root-level model for convenience.
    model.save("dqn_model")

    mean_reward_last20 = float(np.mean(logger.ep_rewards[-20:])) if logger.ep_rewards else 0.0
    mean_episode_len_last20 = float(np.mean(logger.ep_lengths[-20:])) if logger.ep_lengths else 0.0

    metadata = {
        "member": member,
        "experiment_number": exp_num,
        "experiment_name": cfg.experiment_name,
        "tag": tag,
        **asdict(cfg),
        "timesteps": total_timesteps,
        "buffer_size": buffer_size,
        "seed": seed,
        "mean_reward_last20": mean_reward_last20,
        "mean_episode_len_last20": mean_episode_len_last20,
        "total_episodes": len(logger.ep_rewards),
        "model_path": f"{model_dir}/dqn_model.zip",
        "best_model_path": f"{best_dir}/best_model.zip",
        "log_path": log_dir,
    }

    with open(f"{log_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    append_results_csv(metadata)

    print(f"Saved model: {metadata['model_path']}")
    print(f"Saved metadata: {log_dir}/meta.json")
    print(f"Mean reward (last 20 eps): {mean_reward_last20:.2f}")

    train_env.close()
    eval_env.close()
    return metadata


ALL_EXPERIMENTS: dict[int, list[ExperimentConfig]] = {
    1: [
        ExperimentConfig("baseline", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("high_lr", "CnnPolicy", 1e-3, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("low_lr", "CnnPolicy", 5e-5, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("gamma_95", "CnnPolicy", 1e-4, 0.95, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("gamma_999", "CnnPolicy", 1e-4, 0.999, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("batch_64", "CnnPolicy", 1e-4, 0.99, 64, 1.0, 0.01, 0.10),
        ExperimentConfig("batch_16", "CnnPolicy", 1e-4, 0.99, 16, 1.0, 0.01, 0.10),
        ExperimentConfig("eps_end_05", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.05, 0.10),
        ExperimentConfig("slow_decay", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.20),
        ExperimentConfig("mlp_policy", "MlpPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.10),
    ],
    2: [
        ExperimentConfig("lr_2e4", "CnnPolicy", 2e-4, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("gamma_98", "CnnPolicy", 1e-4, 0.98, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("batch_128", "CnnPolicy", 1e-4, 0.99, 128, 1.0, 0.01, 0.10),
        ExperimentConfig("eps_start_05", "CnnPolicy", 1e-4, 0.99, 32, 0.5, 0.01, 0.10),
        ExperimentConfig("eps_end_001", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.001, 0.10),
        ExperimentConfig("lr_5e4", "CnnPolicy", 5e-4, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("combined_A", "CnnPolicy", 1e-4, 0.97, 64, 1.0, 0.01, 0.15),
        ExperimentConfig("combined_B", "CnnPolicy", 2e-4, 0.98, 64, 1.0, 0.01, 0.10),
        ExperimentConfig("fast_decay", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.02, 0.05),
        ExperimentConfig("lr_3e4", "CnnPolicy", 3e-4, 0.99, 32, 1.0, 0.01, 0.10),
    ],
    3: [
        ExperimentConfig("baseline_m3", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("gamma_90", "CnnPolicy", 1e-4, 0.90, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("batch_256", "CnnPolicy", 1e-4, 0.99, 256, 1.0, 0.01, 0.10),
        ExperimentConfig("eps_end_10", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.10, 0.10),
        ExperimentConfig("frac_30", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.30),
        ExperimentConfig("lr_75e5", "CnnPolicy", 7.5e-5, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("gamma_995", "CnnPolicy", 1e-4, 0.995, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("batch_48", "CnnPolicy", 1e-4, 0.99, 48, 1.0, 0.01, 0.10),
        ExperimentConfig("best_combined_m3", "CnnPolicy", 2e-4, 0.995, 64, 1.0, 0.01, 0.12),
        ExperimentConfig("mlp_tuned", "MlpPolicy", 5e-4, 0.99, 64, 1.0, 0.01, 0.10),
    ],
    4: [
        ExperimentConfig("baseline_m4", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("eps_start_08", "CnnPolicy", 1e-4, 0.99, 32, 0.8, 0.01, 0.10),
        ExperimentConfig("frac_08", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.08),
        ExperimentConfig("lr_15e4", "CnnPolicy", 1.5e-4, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("gamma_985", "CnnPolicy", 1e-4, 0.985, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("eps_end_005", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.005, 0.10),
        ExperimentConfig("batch_96", "CnnPolicy", 1e-4, 0.99, 96, 1.0, 0.01, 0.10),
        ExperimentConfig("lr_8e5", "CnnPolicy", 8e-5, 0.99, 32, 1.0, 0.01, 0.10),
        ExperimentConfig("frac_25", "CnnPolicy", 1e-4, 0.99, 32, 1.0, 0.01, 0.25),
        ExperimentConfig("best_combined_m4", "CnnPolicy", 1.5e-4, 0.995, 48, 1.0, 0.005, 0.12),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on ALE/Breakout-v5")
    parser.add_argument("--member", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--exp", type=int, default=None, help="1-based experiment index")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50_000,
        help="Replay buffer size. Lower this if your machine has limited RAM.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    member_exps = ALL_EXPERIMENTS[args.member]
    if args.exp is None:
        selected = list(enumerate(member_exps, start=1))
    else:
        if args.exp < 1 or args.exp > len(member_exps):
            raise ValueError(f"--exp must be between 1 and {len(member_exps)}")
        selected = [(args.exp, member_exps[args.exp - 1])]

    run_summary: list[dict[str, Any]] = []
    for exp_num, cfg in selected:
        metadata = run_experiment(
            member=args.member,
            exp_num=exp_num,
            cfg=cfg,
            total_timesteps=args.timesteps,
            seed=args.seed,
            buffer_size=args.buffer_size,
        )
        run_summary.append(metadata)

    print("\nSUMMARY")
    print("-" * 72)
    for item in run_summary:
        print(
            f"{item['tag']:<40} "
            f"reward(last20)={item['mean_reward_last20']:.2f} "
            f"episodes={item['total_episodes']}"
        )
    print("-" * 72)


if __name__ == "__main__":
    main()
