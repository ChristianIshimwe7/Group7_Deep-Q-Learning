# Formative 3 - Deep Q Learning (ALE/Breakout-v5)

This repository trains and evaluates a DQN agent using Stable Baselines3 and Gymnasium.

## Environment

- Name: `ALE/Breakout-v5`
- Observation: Atari frames wrapped with `AtariWrapper`, stacked with `VecFrameStack(4)`
- Action space: `Discrete(4)`
- Reward: Brick hits increase reward

## Project Files

- `train.py`: Trains DQN and saves models and logs
- `play.py`: Loads a trained model and runs greedy gameplay
- `requirements.txt`: Python dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

### Member 1 quick smoke test

```bash
python train.py --member 1 --exp 1 --timesteps 50000 --seed 42
```

### Member 1 full 10 experiments

```bash
python train.py --member 1 --timesteps 500000 --seed 42
```

### Run one specific experiment

```bash
python train.py --member 1 --exp 3 --timesteps 500000 --seed 42
```

## Automated Member 1 Pipeline (No Manual One-by-One Runs)

This script now supports two modes:

- `full-only`: run selected experiments directly at full timesteps (best for your Kaggle plan)
- `two-stage`: screening first, then rerun top-k at full timesteps

Default behavior is `full-only`.

### Full-only (all 10 experiments, resume-safe)

```bash
python3 run_member1_pipeline.py --mode full-only --member 1 --experiments 1,2,3,4,5,6,7,8,9,10 --full-timesteps 500000 --seed 42 --buffer-size-full 50000 --skip-completed
```

### Two-stage (screening then top-k full)

```bash
python3 run_member1_pipeline.py --mode two-stage --experiments 1,2,3,4,5,6,7,8,9,10 --screening-timesteps 200000 --top-k 3 --full-timesteps 500000 --seed 42 --skip-completed
```

### Quick defaults

```bash
python3 run_member1_pipeline.py
```

Useful variants:

```bash
# Use default full-only mode but custom subset
python3 run_member1_pipeline.py --mode full-only --experiments 3,4,5,6,7,8,9,10 --skip-completed

# Use two-stage mode with custom top-k
python3 run_member1_pipeline.py --mode two-stage --screening-timesteps 150000 --top-k 2
```

### Other members

```bash
python train.py --member 2 --timesteps 500000 --seed 42
python train.py --member 3 --timesteps 500000 --seed 42
python train.py --member 4 --timesteps 500000 --seed 42
```

## Play / Evaluation

Load latest root model (`dqn_model.zip`):

```bash
python play.py --episodes 5
```

Load a specific experiment model:

```bash
python play.py --model models/M1_E01_baseline/dqn_model --episodes 5
```

Headless run:

```bash
python play.py --model models/M1_E01_baseline/dqn_model --no-render --episodes 10
```

## Select Best Model (Member 1)

After your 10 runs are complete, pick the best Member 1 model from
`results/experiments.csv` and copy it to `dqn_model.zip`:

```bash
python3 select_best_model.py --member 1 --metric mean_reward_last20 --output dqn_model.zip
```

Then test the copied model:

```bash
python3 play.py --model dqn_model --episodes 5
```

## Outputs

Each experiment writes:

- Model: `models/<TAG>/dqn_model.zip`
- Best eval model: `best/<TAG>/best_model.zip`
- Eval logs + metadata: `logs/<TAG>/`
- TensorBoard logs: `tensorboard/<TAG>/`
- Aggregated table: `results/experiments.csv`

Example tag format:

- `M1_E01_baseline`

## Hyperparameter Table (Template)

Fill this table with observed behavior from your experiments.

| Member   | Experiment     | policy    |   lr | gamma | batch_size | eps_start | eps_end | eps_fraction | mean_reward_last20 | mean_episode_len_last20 | Behavior notes |
| -------- | -------------- | --------- | ---: | ----: | ---------: | --------: | ------: | -----------: | -----------------: | ----------------------: | -------------- |
| Member 1 | E01_baseline   | CnnPolicy | 1e-4 |  0.99 |         32 |       1.0 |    0.01 |         0.10 |                    |                         |                |
| Member 1 | E02_high_lr    | CnnPolicy | 1e-3 |  0.99 |         32 |       1.0 |    0.01 |         0.10 |                    |                         |                |
| Member 1 | E03_low_lr     | CnnPolicy | 5e-5 |  0.99 |         32 |       1.0 |    0.01 |         0.10 |                    |                         |                |
| Member 1 | E04_gamma_95   | CnnPolicy | 1e-4 |  0.95 |         32 |       1.0 |    0.01 |         0.10 |                    |                         |                |
| Member 1 | E05_gamma_999  | CnnPolicy | 1e-4 | 0.999 |         32 |       1.0 |    0.01 |         0.10 |                    |                         |                |
| Member 1 | E06_batch_64   | CnnPolicy | 1e-4 |  0.99 |         64 |       1.0 |    0.01 |         0.10 |                    |                         |                |
| Member 1 | E07_batch_16   | CnnPolicy | 1e-4 |  0.99 |         16 |       1.0 |    0.01 |         0.10 |                    |                         |                |
| Member 1 | E08_eps_end_05 | CnnPolicy | 1e-4 |  0.99 |         32 |       1.0 |    0.05 |         0.10 |                    |                         |                |
| Member 1 | E09_slow_decay | CnnPolicy | 1e-4 |  0.99 |         32 |       1.0 |    0.01 |         0.20 |                    |                         |                |
| Member 1 | E10_mlp_policy | MlpPolicy | 1e-4 |  0.99 |         32 |       1.0 |    0.01 |         0.10 |                    |                         |                |

## Notes for Presentation

- Compare `CnnPolicy` vs `MlpPolicy` directly (Member 1 E10)
- Use `results/experiments.csv` and per-run `logs/<TAG>/meta.json` for documented evidence
- Record gameplay clip with `play.py` using the best model
