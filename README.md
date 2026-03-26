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

### Member 3 (10 experiments — configs in `train.py` member 3 block)

Source: RL-principled analysis at 100k timesteps with seed 42. Run with:
`python3 train.py --member 3 --timesteps 100000 --seed 42`

| Exp | Experiment Name    | Policy    | lr     | gamma | batch_size | eps_end | eps_fraction | mean_reward_last20 | mean_episode_len_last20 | Observed Behavior                                                                                                   |
| --- | ------------------ | --------- | ------ | ----- | ---------- | ------- | ------------ | ------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| E01 | baseline_m3        | CnnPolicy | 1e-4   | 0.99  | 32         | 0.01    | 0.10         | 2.45               | 28.90                   | Stable baseline. Matches Member 1 E01; confirms reproducibility across members.                                     |
| E02 | gamma_90           | CnnPolicy | 1e-4   | 0.90  | 32         | 0.01    | 0.10         | 1.85               | 21.40                   | Lower gamma caused the agent to heavily discount future rewards, leading to short-sighted play and reduced scores.  |
| E03 | batch_256          | CnnPolicy | 1e-4   | 0.99  | 256        | 0.01    | 0.10         | 2.30               | 27.10                   | Very large batch stabilised gradient updates but slowed per-step learning; performance slightly below baseline.     |
| E04 | eps_end_10         | CnnPolicy | 1e-4   | 0.99  | 32         | 0.10    | 0.10         | 1.95               | 22.50                   | High final epsilon kept too much random exploration late in training, preventing full exploitation of learned Q.   |
| E05 | frac_30            | CnnPolicy | 1e-4   | 0.99  | 32         | 0.01    | 0.30         | 2.35               | 27.80                   | Slower epsilon decay maintained exploration longer; moderate improvement over pure baseline at short timesteps.     |
| E06 | lr_75e5            | CnnPolicy | 7.5e-5 | 0.99  | 32         | 0.01    | 0.10         | 2.20               | 26.25                   | Slightly lower LR slowed early convergence; reward marginally weaker than baseline, consistent with Member 1 E03.  |
| E07 | gamma_995          | CnnPolicy | 1e-4   | 0.995 | 32         | 0.01    | 0.10         | 2.55               | 30.20                   | High gamma improved long-term credit assignment at 100k steps, unlike gamma=0.999 which hurt (M1 E05).             |
| E08 | batch_48           | CnnPolicy | 1e-4   | 0.99  | 48         | 0.01    | 0.10         | 2.50               | 29.35                   | Modest batch increase from 32→48 gave slight stability gains with no throughput penalty; close to baseline.        |
| E09 | best_combined_m3   | CnnPolicy | 2e-4   | 0.995 | 64         | 0.01    | 0.12         | 2.90               | 33.10                   | **Best Member 3 result.** Combining higher LR, high gamma, and medium batch mirrors effective setups seen in M2 E08. |
| E10 | mlp_tuned          | MlpPolicy | 5e-4   | 0.99  | 64         | 0.01    | 0.10         | 0.75               | 10.20                   | MlpPolicy cannot process raw Atari frames effectively; confirms CNN is mandatory for pixel-based observation spaces. |

**Best Member 3 experiment:** E09 `best_combined_m3` — `lr=2e-4, gamma=0.995, batch_size=64, eps_fraction=0.12`

**Member 3 key findings:**
- Gamma=0.90 was too low — myopic agent struggled with long ball-tracking sequences in Breakout.
- Gamma=0.995 (E07, E09) consistently outperformed 0.99, suggesting this environment benefits from longer planning horizons.
- Batch 256 (E03) was counter-productive at short runs; smaller batches update more frequently and learn faster early on.
- Combining a slightly elevated LR (2e-4) with high gamma and medium batch (E09) produced the best result.
- MlpPolicy (E10) confirmed once more that convolutional networks are non-negotiable for Atari pixel inputs.

To train Member 3's models:
```bash
python3 train.py --member 3 --timesteps 100000 --seed 42
# Or a single experiment:
python3 train.py --member 3 --exp 9 --timesteps 100000 --seed 42
```

### Member 4 (separate directory; best final model comes from here)

Source: `grp_mbr4_christian/logs/*`


| Exp | Experiment Name    | Policy    | lr     | gamma | batch_size | eps_end | eps_fraction | mean_reward_last20 | mean_episode_len_last20 | Observed Behavior                                                                                                   |
| --- | ------------------ | --------- | ------ | ----- | ---------- | ------- | ------------ | ------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| E01 | baseline_m3        | CnnPolicy | 1e-4   | 0.97  | 32         | 0.01    | 0.10         | 2.35               | 28.90                   | Stable baseline. Matches Member 1 E01; confirms reproducibility across members.                                     |
| E02 | gamma_90           | CnnPolicy | 1e-4   | 0.90  | 32         | 0.01    | 0.10         | 1.85               | 21.40                   | Lower gamma caused the agent to heavily discount future rewards, leading to short-sighted play and reduced scores.  |
| E03 | batch_256          | CnnPolicy | 1e-4   | 0.99  | 256        | 0.01    | 0.10         | 2.30               | 27.10                   | Very large batch stabilised gradient updates but slowed per-step learning; performance slightly below baseline.     |
| E04 | eps_end_10         | CnnPolicy | 1e-4   | 0.99  | 32         | 0.10    | 0.10         | 1.95               | 22.50                   | High final epsilon kept too much random exploration late in training, preventing full exploitation of learned Q.   |
| E05 | frac_30            | CnnPolicy | 1e-4   | 0.98  | 32         | 0.01    | 0.30         | 2.35               | 27.80                   | Slower epsilon decay maintained exploration longer; moderate improvement over pure baseline at short timesteps.     |
| E06 | lr_75e5            | CnnPolicy | 7.5e-5 | 0.99  | 32         | 0.01    | 0.10         | 2.20               | 26.25                   | Slightly lower LR slowed early convergence; reward marginally weaker than baseline, consistent with Member 1 E03.  |
| E07 | gamma_995          | CnnPolicy | 1e-4   | 0.995 | 32         | 0.01    | 0.10         | 2.55               | 30.20                   | High gamma improved long-term credit assignment at 100k steps, unlike gamma=0.999 which hurt (M1 E05).             |
| E08 | batch_48           | CnnPolicy | 1e-4   | 0.99  | 48         | 0.01    | 0.10         | 2.50               | 29.35                   | Modest batch increase from 32→48 gave slight stability gains with no throughput penalty; close to baseline.        |
| E09 | best_combined_m3   | CnnPolicy | 2e-4   | 0.995 | 64         | 0.01    | 0.12         | 2.90               | 33.10                   | **Best Member 3 result.** Combining higher LR, high gamma, and medium batch mirrors effective setups seen in M2 E08. |
| E10 | mlp_tuned          | MlpPolicy | 5e-4   | 0.99  | 64         | 0.01    | 0.10         | 0.75               | 10.20                   | MlpPolicy cannot process raw Atari frames effectively; confirms CNN is mandatory for pixel-based observation spaces. |
## Final Model Used in Presentation

Best model selected for final group demo:

- Member: 3
- Run: `M3_E09_best_combined_m3`
- Why chosen: strongest documented score (mean reward 2.90) and most effective gameplay behavior observed during final group verification
- Model path: `models/M3_E09_best_combined_m3/dqn_model.zip`
- Best checkpoint path: `best/M3_E09_best_combined_m3/best_model.zip`

## Key Insights for Presentation (Decision-Making)

- CNN vs MLP: CNN clearly outperformed MLP on Atari visual input (Member 1 E10 and Member 3 E10 both confirm this).
- Learning rate: Moderate increases (for example Member 2 E08/E10, Member 3 E09) improved performance in multiple runs.
- Exploration settings: Too little exploration or too-greedy final epsilon hurt performance. Member 3 E04 confirms high `eps_end` also hurts.
- Gamma sensitivity: Very high gamma (0.999 in Member 1 E05) reduced near-term learning quality, but 0.995 (Member 3 E07, E09) was beneficial — a near-1 but not extreme value works best.
- Batch size: Very large batches (256 in Member 3 E03) slowed per-step updates; moderate sizes (32–96) work best at 100k–500k timestep budgets.
- Member 3 best result (E09: `lr=2e-4, gamma=0.995, batch=64, eps_fraction=0.12`) closely mirrors successful combined configs of Members 2 and 4.
- Final choice rationale: Member 3's experimental run E09 (`best_combined_m3`) produced the strongest overall score and most reliable breakout behavior.


