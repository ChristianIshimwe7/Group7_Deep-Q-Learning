# Formative 3 - Deep Q Learning (ALE/Breakout-v5)

This repository contains training and evaluation code for a Deep Q-Network (DQN) agent using Stable Baselines3 and Gymnasium, with experiment tracking for group members.

## Assignment Coverage Checklist

- [x] Atari environment selected: `ALE/Breakout-v5`
- [x] `train.py` implemented for DQN training and model saving
- [x] `play.py` implemented for loading model and greedy gameplay
- [x] MLP vs CNN comparison included (Member 1 experiment E10 uses `MlpPolicy`)
- [x] Hyperparameter tuning documented with experiment table(s)
- [x] Best-performing model identified for final demo
- [x] Gameplay video link included

## Environment and RL Setup

- Environment: `ALE/Breakout-v5`
- Wrapper pipeline: `AtariWrapper` + `VecFrameStack(4)`
- Agent: `DQN` (Stable Baselines3)
- Evaluation policy: Greedy Q-policy (`deterministic=True` in `model.predict`)

Gameplay / script demo video (used in presentation):

- https://www.youtube.com/shorts/SixaEsx-IwY

## Repository Layout

- Root folder contains the shared group pipeline and Member 1/2 outputs.
- Member 4 used a separate working directory for training and logs:
	- `grp_mbr4_christian/`

This is intentional and documented here so reviewers can find all artifacts.

## Main Files

- `train.py`: Shared training script with multi-member experiment presets.
- `play.py`: Shared evaluation script with greedy action selection.
- `run_member1_pipeline.py`: Optional automation script for running Member 1 experiments.
- `select_best_model.py`: Utility to pick best model from tracked experiments.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Train

### Shared root experiments (Members 1-3 and configured Member 4 list)

```bash
# Run all experiments configured for one member
python3 train.py --member 1 --timesteps 100000 --seed 42

# Run one specific experiment
python3 train.py --member 1 --exp 10 --timesteps 100000 --seed 42
```

### Member 4 separate folder experiments

```bash
cd grp_mbr4_christian
python3 train.py --exp 1 --timesteps 500000
python3 train.py --exp 2 --timesteps 500000
```

## How to Play (Evaluation)

### Root script usage

```bash
python3 play.py --episodes 5
python3 play.py --model models/M1_E01_baseline/dqn_model --episodes 5
python3 play.py --model models/M1_E01_baseline/dqn_model --no-render --episodes 10
```

### Play Member 4 best model (separate directory)

```bash
cd grp_mbr4_christian
python3 play.py --model models/M4_E01_baseline_m4/dqn_model --episodes 5
```

## Output Artifacts

Each run generates:

- Trained model: `models/<TAG>/dqn_model.zip`
- Best checkpoint (eval callback): `best/<TAG>/best_model.zip`
- Run metadata: `logs/<TAG>/meta.json` (when available)
- Evaluation arrays: `logs/<TAG>/evaluations.npz`
- TensorBoard logs: `tensorboard/<TAG>/`
- Aggregate results CSV: `results/experiments.csv`

## Hyperparameter Tuning Results

### Member 1 (10 experiments completed)

Source: `results/experiments.csv`

| Member   | Experiment     | Policy    | lr   | gamma | batch_size | eps_start | eps_end | eps_fraction | timesteps | mean_reward_last20 | mean_episode_len_last20 | Behavior notes                                             |
| -------- | -------------- | --------- | ---- | ----- | ---------- | --------- | ------- | ------------ | --------- | ------------------ | ----------------------- | ---------------------------------------------------------- |
| Member 1 | E01_baseline   | CnnPolicy | 1e-4 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.45               | 28.90                   | Stable baseline.                                           |
| Member 1 | E02_high_lr    | CnnPolicy | 1e-3 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.45               | 26.10                   | Similar reward to baseline, shorter episodes.              |
| Member 1 | E03_low_lr     | CnnPolicy | 5e-5 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.20               | 26.25                   | Slower learning.                                           |
| Member 1 | E04_gamma_95   | CnnPolicy | 1e-4 | 0.95  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.45               | 27.65                   | Close to baseline.                                         |
| Member 1 | E05_gamma_999  | CnnPolicy | 1e-4 | 0.999 | 32         | 1.0       | 0.01    | 0.10         | 100000    | 1.75               | 22.30                   | Performance dropped with very high gamma.                  |
| Member 1 | E06_batch_64   | CnnPolicy | 1e-4 | 0.99  | 64         | 1.0       | 0.01    | 0.10         | 100000    | 2.45               | 27.00                   | Stable; no clear gain over baseline.                       |
| Member 1 | E07_batch_16   | CnnPolicy | 1e-4 | 0.99  | 16         | 1.0       | 0.01    | 0.10         | 100000    | 2.10               | 25.15                   | Slightly noisier and weaker.                               |
| Member 1 | E08_eps_end_05 | CnnPolicy | 1e-4 | 0.99  | 32         | 1.0       | 0.05    | 0.10         | 100000    | 2.00               | 22.65                   | More exploration late in training reduced score.           |
| Member 1 | E09_slow_decay | CnnPolicy | 1e-4 | 0.99  | 32         | 1.0       | 0.01    | 0.20         | 100000    | 2.25               | 25.00                   | Slower epsilon decay gave moderate results.                |
| Member 1 | E10_mlp_policy | MlpPolicy | 1e-4 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 0.70               | 9.70                    | MLP underperformed strongly vs CNN for image observations. |

### Member 2 (10 experiments completed)

Source: `results/experiments.csv`

| Member   | Experiment       | Policy    | lr   | gamma | batch_size | eps_start | eps_end | eps_fraction | timesteps | mean_reward_last20 | mean_episode_len_last20 | Behavior notes                                 |
| -------- | ---------------- | --------- | ---- | ----- | ---------- | --------- | ------- | ------------ | --------- | ------------------ | ----------------------- | ---------------------------------------------- |
| Member 2 | E01_lr_2e4       | CnnPolicy | 2e-4 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.75               | 31.75                   | Strong baseline variant.                       |
| Member 2 | E02_gamma_98     | CnnPolicy | 1e-4 | 0.98  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.50               | 27.85                   | Stable, slightly weaker than E01.              |
| Member 2 | E03_batch_128    | CnnPolicy | 1e-4 | 0.99  | 128        | 1.0       | 0.01    | 0.10         | 100000    | 2.55               | 28.65                   | Stable with larger batch.                      |
| Member 2 | E04_eps_start_05 | CnnPolicy | 1e-4 | 0.99  | 32         | 0.5       | 0.01    | 0.10         | 100000    | 1.70               | 19.50                   | Lower initial exploration hurt early learning. |
| Member 2 | E05_eps_end_001  | CnnPolicy | 1e-4 | 0.99  | 32         | 1.0       | 0.001   | 0.10         | 100000    | 1.65               | 19.60                   | Too-greedy end phase hurt performance.         |
| Member 2 | E06_lr_5e4       | CnnPolicy | 5e-4 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.90               | 32.80                   | Faster learning; high-performing setup.        |
| Member 2 | E07_combined_A   | CnnPolicy | 1e-4 | 0.97  | 64         | 1.0       | 0.01    | 0.15         | 100000    | 1.50               | 18.75                   | Weakest combined configuration.                |
| Member 2 | E08_combined_B   | CnnPolicy | 2e-4 | 0.98  | 64         | 1.0       | 0.01    | 0.10         | 100000    | 3.30               | 35.75                   | Best Member 2 result.                          |
| Member 2 | E09_fast_decay   | CnnPolicy | 1e-4 | 0.99  | 32         | 1.0       | 0.02    | 0.05         | 100000    | 2.15               | 25.10                   | Mixed impact from faster decay.                |
| Member 2 | E10_lr_3e4       | CnnPolicy | 3e-4 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 3.15               | 34.35                   | Second-best Member 2 result.                   |

Extended run recorded:

| Member   | Experiment | timesteps | mean_reward_last20 |
| -------- | ---------- | --------- | ------------------ |
| Member 2 | E01_lr_2e4 | 500000    | 2.40               |

### Member 4 (separate directory; best final model comes from here)

Source: `grp_mbr4_christian/logs/*`

| Member   | Experiment          | Policy    | lr   | gamma | batch_size | eps_start | eps_end | eps_fraction | timesteps            | mean_reward_last20                          | mean_episode_len_last20  | Evidence                                                                   |
| -------- | ------------------- | --------- | ---- | ----- | ---------- | --------- | ------- | ------------ | -------------------- | ------------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| Member 4 | M4_E01_baseline_m4  | CnnPolicy | 1e-4 | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 500000               | 3.45 (train meta) / 2.69 (eval npz summary) | 28.89 (eval npz summary) | `grp_mbr4_christian/logs/M4_E01_baseline_m4/meta.json` + `evaluations.npz` |
| Member 4 | M4_E02_eps_start_08 | CnnPolicy | 1e-4 | 0.99  | 32         | 0.8       | 0.01    | 0.10         | not recorded in meta | 1.60 (eval npz summary, 4 eval points)      | 18.45 (eval npz summary) | `grp_mbr4_christian/logs/M4_E02_eps_start_08/evaluations.npz`              |

## Final Model Used in Presentation

Best model selected for final group demo:

- Member: 4
- Run: `M4_E01_baseline_m4`
- Why chosen: strongest available score among documented runs and best gameplay behavior during testing
- Model path: `grp_mbr4_christian/models/M4_E01_baseline_m4/dqn_model.zip`
- Best checkpoint path: `grp_mbr4_christian/best/M4_E01_baseline_m4/best_model.zip`

## Key Insights for Presentation (Decision-Making)

- CNN vs MLP: CNN clearly outperformed MLP on Atari visual input (Member 1 E10 underperformed).
- Learning rate: Moderate increases (for example Member 2 E08/E10 settings) improved performance in multiple runs.
- Exploration settings: Too little exploration or too-greedy final epsilon hurt performance.
- Gamma sensitivity: Very high gamma (0.999 in Member 1 E05) reduced near-term learning quality.
- Final choice rationale: Member 4 baseline run produced the strongest practical gameplay for final demo.

