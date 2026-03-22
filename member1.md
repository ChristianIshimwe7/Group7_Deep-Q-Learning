Run all experiments at 100k (fast comparison pass).
Run only the first and last experiments at 500k (deeper check).
Keep everything resume-safe with --skip-completed.


# Step 1: all 10 experiments at 100k
python3 run_member1_pipeline.py \
  --mode full-only \
  --experiments 1,2,3,4,5,6,7,8,9,10 \
  --full-timesteps 100000 \
  --buffer-size-full 30000 \
  --seed 42 \
  --skip-completed

# Step 2: only first and last at 500k
python3 run_member1_pipeline.py \
  --mode full-only \
  --experiments 1,10 \
  --full-timesteps 500000 \
  --buffer-size-full 50000 \
  --seed 42 \
  --skip-completed


Then pick best model from the full runs (500k) if that is your final submission candidate:



  python3 select_best_model.py --member 1 --metric mean_reward_last20 --output dqn_model.zip