"""
run_member1_pipeline.py
=======================
Automate Member experiments without manually running one-by-one.

Modes:
1) full-only: run selected experiments directly at full timesteps.
2) two-stage: run screening first, then full runs for top-k experiments.

The script reads results/experiments.csv to optionally skip completed runs,
which is useful when resuming after notebook/session interruptions.

Examples
--------
# Run all 10 experiments fully (default)
python3 run_member1_pipeline.py

# Resume-safe full runs (skip any already completed in CSV)
python3 run_member1_pipeline.py --mode full-only --skip-completed

# Two-stage mode
python3 run_member1_pipeline.py --mode two-stage --screening-timesteps 200000 --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


RESULTS_CSV = Path("results/experiments.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Member 1 experiments end-to-end")
    parser.add_argument("--member", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--mode",
        type=str,
        default="full-only",
        choices=["full-only", "two-stage"],
        help="full-only runs all selected experiments at full timesteps; two-stage does screening then top-k full runs",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated experiment numbers to process",
    )
    parser.add_argument("--screening-timesteps", type=int, default=150_000)
    parser.add_argument("--full-timesteps", type=int, default=500_000)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size-screening", type=int, default=30_000)
    parser.add_argument("--buffer-size-full", type=int, default=50_000)
    parser.add_argument("--metric", type=str, default="mean_reward_last20")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip runs that already exist in results/experiments.csv for the same member/exp/timesteps",
    )
    return parser.parse_args()


def parse_experiment_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("No experiment numbers provided")
    for v in values:
        if v < 1 or v > 10:
            raise ValueError("Experiment numbers must be between 1 and 10")
    return values


def run_train(member: int, exp: int, timesteps: int, seed: int, buffer_size: int) -> None:
    cmd = [
        sys.executable,
        "train.py",
        "--member",
        str(member),
        "--exp",
        str(exp),
        "--timesteps",
        str(timesteps),
        "--seed",
        str(seed),
        "--buffer-size",
        str(buffer_size),
    ]
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_rows() -> list[dict[str, str]]:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_row_for(rows: list[dict[str, str]], member: int, exp: int, timesteps: int) -> dict[str, str] | None:
    # Latest row is the last matching row in CSV append order.
    for row in reversed(rows):
        if str(row.get("member", "")).strip() != str(member):
            continue
        if str(row.get("experiment_number", "")).strip() != str(exp):
            continue
        if str(row.get("timesteps", "")).strip() != str(timesteps):
            continue
        return row
    return None


def is_completed(rows: list[dict[str, str]], member: int, exp: int, timesteps: int) -> bool:
    return latest_row_for(rows, member=member, exp=exp, timesteps=timesteps) is not None


def run_many(
    *,
    rows: list[dict[str, str]],
    member: int,
    experiments: list[int],
    timesteps: int,
    seed: int,
    buffer_size: int,
    skip_completed: bool,
    stage_name: str,
) -> None:
    print("\n" + "=" * 72)
    print(stage_name)
    print(f"Member={member} experiments={experiments}")
    print(f"timesteps={timesteps} buffer={buffer_size} skip_completed={skip_completed}")
    print("=" * 72)

    for exp in experiments:
        if skip_completed and is_completed(rows, member=member, exp=exp, timesteps=timesteps):
            print(f"Skipping E{exp:02d} (already completed at {timesteps} timesteps)")
            continue

        run_train(
            member=member,
            exp=exp,
            timesteps=timesteps,
            seed=seed,
            buffer_size=buffer_size,
        )

        # Refresh CSV rows after each completed run for resume-safe behavior.
        rows[:] = read_rows()


def select_top_k(
    rows: list[dict[str, str]],
    member: int,
    experiments: list[int],
    screening_timesteps: int,
    metric: str,
    top_k: int,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for exp in experiments:
        row = latest_row_for(rows, member=member, exp=exp, timesteps=screening_timesteps)
        if row is None:
            continue
        try:
            row["_metric_value"] = str(float(row.get(metric, "")))
            candidates.append(row)
        except ValueError:
            continue

    candidates.sort(key=lambda r: float(r["_metric_value"]), reverse=True)
    return candidates[:top_k]


def main() -> None:
    args = parse_args()
    experiments = parse_experiment_list(args.experiments)
    rows = read_rows()

    if args.mode == "full-only":
        run_many(
            rows=rows,
            member=args.member,
            experiments=experiments,
            timesteps=args.full_timesteps,
            seed=args.seed,
            buffer_size=args.buffer_size_full,
            skip_completed=args.skip_completed,
            stage_name="Full-only pipeline",
        )

        print("\nPipeline complete.")
        print("Run this next to copy best full model:")
        print(f"{sys.executable} select_best_model.py --member {args.member} --metric mean_reward_last20")
        return

    run_many(
        rows=rows,
        member=args.member,
        experiments=experiments,
        timesteps=args.screening_timesteps,
        seed=args.seed,
        buffer_size=args.buffer_size_screening,
        skip_completed=args.skip_completed,
        stage_name="Stage 1: Screening",
    )

    rows = read_rows()
    top_rows = select_top_k(
        rows=rows,
        member=args.member,
        experiments=experiments,
        screening_timesteps=args.screening_timesteps,
        metric=args.metric,
        top_k=args.top_k,
    )

    if not top_rows:
        raise RuntimeError("No screening results found to select top-k experiments")

    top_exps = [int(r["experiment_number"]) for r in top_rows]

    print("\n" + "=" * 72)
    print("Top experiments from screening")
    for idx, row in enumerate(top_rows, start=1):
        print(
            f"{idx}. E{int(row['experiment_number']):02d}_{row.get('experiment_name','')} "
            f"{args.metric}={float(row['_metric_value']):.2f}"
        )
    print("=" * 72)

    print("\n" + "=" * 72)
    print("Stage 2 target experiments:", top_exps)
    print("=" * 72)

    run_many(
        rows=rows,
        member=args.member,
        experiments=top_exps,
        timesteps=args.full_timesteps,
        seed=args.seed,
        buffer_size=args.buffer_size_full,
        skip_completed=args.skip_completed,
        stage_name="Stage 2: Full training for selected top-k",
    )

    print("\nPipeline complete.")
    print("Run this next to copy best full model:")
    print(f"{sys.executable} select_best_model.py --member {args.member} --metric mean_reward_last20")


if __name__ == "__main__":
    main()
