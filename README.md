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

## Hyperparameter Table 

Table filled with observed behavior from member's experiments.

Member 1 experiments
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

Member 2 experiments

All experiments were initially run at 100,000 timesteps to compare configurations efficiently. Selected experiments were extended to 500,000 timesteps for deeper evaluation.

| Member   | Experiment        | Policy    | lr    | gamma | batch_size | eps_start | eps_end | eps_fraction | timesteps | mean_reward_last20 | mean_episode_len_last20 | Behavior notes |
|----------|-------------------|-----------|-------|-------|------------|-----------|---------|--------------|-----------|--------------------|--------------------------|----------------|
| Member 2 | E01_lr_2e4        | CnnPolicy | 2e-4  | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.75               | 31.8                     | Steady improvement; stronger than default learning rate. |
| Member 2 | E02_gamma_98      | CnnPolicy | 1e-4  | 0.98  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.50               | 27.9                     | Stable but slightly weaker than E01. |
| Member 2 | E03_batch_128     | CnnPolicy | 1e-4  | 0.99  | 128        | 1.0       | 0.01    | 0.10         | 100000    | 2.55               | 28.6                     | Larger batch gave fairly stable learning. |
| Member 2 | E04_eps_start_05  | CnnPolicy | 1e-4  | 0.99  | 32         | 0.5       | 0.01    | 0.10         | 100000    | 1.70               | 19.5                     | Lower starting exploration hurt performance. |
| Member 2 | E05_eps_end_001   | CnnPolicy | 1e-4  | 0.99  | 32         | 1.0       | 0.001   | 0.10         | 100000    | 1.65               | 19.6                     | Very low final exploration reduced performance. |
| Member 2 | E06_lr_5e4        | CnnPolicy | 5e-4  | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 2.90               | 32.8                     | Faster learning rate performed well. |
| Member 2 | E07_combined_A    | CnnPolicy | 1e-4  | 0.97  | 64         | 1.0       | 0.01    | 0.15         | 100000    | 1.50               | 18.8                     | This combination was the weakest overall. |
| Member 2 | E08_combined_B    | CnnPolicy | 2e-4  | 0.98  | 64         | 1.0       | 0.01    | 0.10         | 100000    | 3.30               | 35.8                     | Best overall; strongest reward and longer survival. |
| Member 2 | E09_fast_decay    | CnnPolicy | 1e-4  | 0.99  | 32         | 1.0       | 0.02    | 0.05         | 100000    | 2.15               | 25.1                     | Faster decay gave mixed results. |
| Member 2 | E10_lr_3e4        | CnnPolicy | 3e-4  | 0.99  | 32         | 1.0       | 0.01    | 0.10         | 100000    | 3.15               | 34.4                     | Very strong result; second best after E08. |

### Extended Run (500k Timesteps)

| Member   | Experiment | Timesteps | mean_reward_last20 | Notes |
|----------|------------|-----------|--------------------|-------|
| Member 2 | E01_lr_2e4 | 500000    | 2.40               | Full-length run completed separately for deeper evaluation. |

## Key Insights — Member 2

The best performing experiment was E08_combined_B, followed by E10_lr_3e4 and E06_lr_5e4. These configurations showed that moderately higher learning rates and balanced parameter combinations improved performance.

Experiments with reduced exploration, such as E04 and E05, performed worse, showing that exploration is important for learning in early stages.

Overall, learning rate and combined parameter tuning had the biggest impact on performance.

## Notes for Presentation

- Compare `CnnPolicy` vs `MlpPolicy` directly (Member 1 E10)
- Use `results/experiments.csv` and per-run `logs/<TAG>/meta.json` for documented evidence
- Record gameplay clip with `play.py` using the best model
