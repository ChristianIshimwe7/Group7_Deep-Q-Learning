"""
train.py  —  MEMBER 4
=====================
Member 4's 10 DQN experiments on ALE/Breakout-v5.
Focus: Eps Start, Decay Fraction, Moderate LR, Gamma, Batch Range, Combined Best

Usage
-----
    python train.py                        # run all 10 experiments
    python train.py --exp 1               # run only experiment 1
    python train.py --exp 1 --timesteps 50000   # quick test
"""

import os
import json
import argparse
import numpy as np
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

gym.register_envs(ale_py)

MEMBER = 4

# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
class TrainingLogger(BaseCallback):
    def __init__(self, log_interval=10_000):
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards = []
        self.ep_lengths = []
        self._cur_rew   = {}
        self._cur_len   = {}

    def _on_step(self) -> bool:
        for i, (done, rew) in enumerate(
            zip(self.locals["dones"], self.locals["rewards"])
        ):
            self._cur_rew[i] = self._cur_rew.get(i, 0.0) + float(rew)
            self._cur_len[i] = self._cur_len.get(i, 0)   + 1
            if done:
                self.ep_rewards.append(self._cur_rew[i])
                self.ep_lengths.append(self._cur_len[i])
                self._cur_rew[i] = 0.0
                self._cur_len[i] = 0
        if self.n_calls % self.log_interval == 0 and self.ep_rewards:
            print(
                f"  [step {self.num_timesteps:>8,}]  "
                f"mean_reward={np.mean(self.ep_rewards[-20:]):.2f}  "
                f"total_eps={len(self.ep_rewards)}"
            )
        return True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def make_env_fn(render_mode=None):
    def _init():
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        env = AtariWrapper(env)
        return env
    return _init

def build_env(render_mode=None, n_stack=4):
    return VecFrameStack(DummyVecEnv([make_env_fn(render_mode)]), n_stack=n_stack)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_experiment(
    exp_num, experiment_name,
    policy="CnnPolicy",
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.10,
    total_timesteps=500_000,
):
    tag        = f"M{MEMBER}_E{exp_num:02d}_{experiment_name}"
    model_path = f"models/{tag}"
    best_path  = f"best/{tag}"
    log_path   = f"logs/{tag}"

    for d in [model_path, best_path, log_path]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  MEMBER {MEMBER}  |  Experiment {exp_num:02d}  |  {experiment_name}")
    print(f"  policy={policy}  lr={learning_rate}  gamma={gamma}  batch={batch_size}")
    print(f"  eps: {exploration_initial_eps} -> {exploration_final_eps} "
          f"over {exploration_fraction*100:.0f}% of training")
    print(f"{'='*65}\n")

    train_env = build_env()
    eval_env  = build_env()

    model = DQN(
        policy                  = policy,
        env                     = train_env,
        learning_rate           = learning_rate,
        buffer_size             = 100_000,
        learning_starts         = 10_000,
        batch_size              = batch_size,
        gamma                   = gamma,
        train_freq              = 4,
        target_update_interval  = 1_000,
        exploration_fraction    = exploration_fraction,
        exploration_initial_eps = exploration_initial_eps,
        exploration_final_eps   = exploration_final_eps,
        optimize_memory_usage   = False,
        verbose                 = 0,
        tensorboard_log         = f"tensorboard/{tag}",
    )

    logger  = TrainingLogger()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = best_path,
        log_path             = log_path,
        eval_freq            = 25_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
    )

    model.learn(total_timesteps=total_timesteps,
                callback=[logger, eval_cb], progress_bar=True)

    model.save(f"{model_path}/dqn_model")
    model.save("dqn_model")

    mean_rew = float(np.mean(logger.ep_rewards[-20:])) if logger.ep_rewards else 0.0

    with open(f"{log_path}/meta.json", "w") as f:
        json.dump(dict(
            tag=tag, policy=policy,
            learning_rate=learning_rate, gamma=gamma, batch_size=batch_size,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            total_timesteps=total_timesteps,
            mean_reward_last20=mean_rew,
            total_episodes=len(logger.ep_rewards),
        ), f, indent=2)

    print(f"\n  Model saved  -> {model_path}/dqn_model.zip")
    print(f"  Mean reward (last 20 eps): {mean_rew:.2f}\n")

    train_env.close()
    eval_env.close()
    return mean_rew


# ---------------------------------------------------------------------------
# MEMBER 4 — 10 Experiments
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # E01 — Baseline replica (reference point)
    dict(experiment_name="baseline_m4",
         policy="CnnPolicy", learning_rate=1e-4, gamma=0.99,
         batch_size=32, exploration_initial_eps=1.0,
         exploration_final_eps=0.01, exploration_fraction=0.10),

    # E02 — Lower starting epsilon (agent starts semi-greedy)
    dict(experiment_name="eps_start_08",
         policy="CnnPolicy", learning_rate=1e-4, gamma=0.99,
         batch_size=32, exploration_initial_eps=0.8,
         exploration_final_eps=0.01, exploration_fraction=0.10),

    # E03 — Faster epsilon decay (8% fraction instead of 10%)
    dict(experiment_name="frac_08",
         policy="CnnPolicy", learning_rate=1e-4, gamma=0.99,
         batch_size=32, exploration_initial_eps=1.0,
         exploration_final_eps=0.01, exploration_fraction=0.08),

    # E04 — Moderate lr increase to 1.5e-4 (halfway between 1e-4 and 2e-4)
    dict(experiment_name="lr_15e4",
         policy="CnnPolicy", learning_rate=1.5e-4, gamma=0.99,
         batch_size=32, exploration_initial_eps=1.0,
         exploration_final_eps=0.01, exploration_fraction=0.10),

    # E05 — Gamma 0.985 (subtle reduction from 0.99)
    dict(experiment_name="gamma_985",
         policy="CnnPolicy", learning_rate=1e-4, gamma=0.985,
         batch_size=32, exploration_initial_eps=1.0,
         exploration_final_eps=0.01, exploration_fraction=0.10),

    # E06 — Very low final epsilon 0.005 (more greedy at the end)
    dict(experiment_name="eps_end_005",
         policy="CnnPolicy", learning_rate=1e-4, gamma=0.99,
         batch_size=32, exploration_initial_eps=1.0,
         exploration_final_eps=0.005, exploration_fraction=0.10),

    # E07 — Large batch 96 (very smooth but slow gradient updates)
    dict(experiment_name="batch_96",
         policy="CnnPolicy", learning_rate=1e-4, gamma=0.99,
         batch_size=96, exploration_initial_eps=1.0,
         exploration_final_eps=0.01, exploration_fraction=0.10),

    # E08 — Very low lr 8e-5 (slower than baseline)
    dict(experiment_name="lr_8e5",
         policy="CnnPolicy", learning_rate=8e-5, gamma=0.99,
         batch_size=32, exploration_initial_eps=1.0,
         exploration_final_eps=0.01, exploration_fraction=0.10),

    # E09 — Long exploration fraction 25%
    dict(experiment_name="frac_25",
         policy="CnnPolicy", learning_rate=1e-4, gamma=0.99,
         batch_size=32, exploration_initial_eps=1.0,
         exploration_final_eps=0.01, exploration_fraction=0.25),

    # E10 — Best combined (Member 4's optimal guess)
    dict(experiment_name="best_combined_m4",
         policy="CnnPolicy", learning_rate=1.5e-4, gamma=0.995,
         batch_size=48, exploration_initial_eps=1.0,
         exploration_final_eps=0.005, exploration_fraction=0.12),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Member 4 DQN Experiments")
    parser.add_argument("--exp",       type=int, default=None,
                        help="Run only experiment N (1-10). Omit to run all.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    if args.exp is not None:
        experiments = [(args.exp, EXPERIMENTS[args.exp - 1])]
    else:
        experiments = [(i+1, e) for i, e in enumerate(EXPERIMENTS)]

    results = {}
    for num, cfg in experiments:
        r = run_experiment(exp_num=num, total_timesteps=args.timesteps, **cfg)
        results[f"E{num:02d}_{cfg['experiment_name']}"] = r

    print("\n" + "="*65)
    print(f"  SUMMARY  —  Member {MEMBER}")
    print("="*65)
    for name, r in results.items():
        print(f"  {name:<40}  mean_reward={r:.2f}")
    print("="*65)
