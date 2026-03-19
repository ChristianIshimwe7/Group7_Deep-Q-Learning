"""
select_best_model.py
====================
Select the best experiment from results/experiments.csv and copy its model
as a submission-ready file (default: dqn_model.zip).

Examples
--------
python3 select_best_model.py --member 1
python3 select_best_model.py --member 1 --metric mean_reward_last20 --output dqn_model.zip
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best trained model from CSV results")
    parser.add_argument("--csv", default="results/experiments.csv", help="Path to experiment CSV")
    parser.add_argument("--member", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--metric", default="mean_reward_last20", help="Numeric metric column to rank")
    parser.add_argument("--output", default="dqn_model.zip", help="Output model path")
    parser.add_argument(
        "--mode",
        default="max",
        choices=["max", "min"],
        help="Use max for metrics where higher is better, min where lower is better",
    )
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_model_zip(model_path_value: str) -> Path:
    path = Path(model_path_value)
    if path.suffix == ".zip":
        return path
    return path.with_suffix(".zip")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    member_rows: list[dict[str, Any]] = [
        row for row in rows if str(row.get("member", "")).strip() == str(args.member)
    ]
    if not member_rows:
        raise ValueError(f"No rows found for member {args.member} in {csv_path}")

    valid_rows: list[dict[str, Any]] = []
    for row in member_rows:
        metric_val = to_float(row.get(args.metric))
        if metric_val is None:
            continue
        row["_metric_val"] = metric_val
        valid_rows.append(row)

    if not valid_rows:
        raise ValueError(
            f"No valid numeric values for metric '{args.metric}' for member {args.member}"
        )

    reverse = args.mode == "max"
    valid_rows.sort(key=lambda r: r["_metric_val"], reverse=reverse)
    best = valid_rows[0]

    model_path_raw = best.get("model_path", "").strip()
    if not model_path_raw:
        raise ValueError("Best row has empty model_path")

    model_zip = resolve_model_zip(model_path_raw)
    if not model_zip.exists():
        raise FileNotFoundError(
            f"Model file not found on disk: {model_zip}\n"
            "Train may not have finished, or paths in CSV are stale."
        )

    output_path = Path(args.output)
    output_dir = output_path.parent
    if output_dir and str(output_dir) != ".":
        output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_zip, output_path)

    print("Best run selected")
    print("-" * 60)
    print(f"member: {best.get('member')}")
    print(f"experiment: E{best.get('experiment_number')}_{best.get('experiment_name')}")
    print(f"tag: {best.get('tag')}")
    print(f"metric: {args.metric}={best.get('_metric_val')}")
    print(f"source model: {model_zip}")
    print(f"copied to: {output_path}")
    print("-" * 60)

    print("Top 5 runs for this member")
    print("-" * 60)
    for idx, row in enumerate(valid_rows[:5], start=1):
        print(
            f"{idx}. {row.get('tag','')}  "
            f"{args.metric}={row.get('_metric_val')}  "
            f"model={row.get('model_path','')}"
        )


if __name__ == "__main__":
    main()
