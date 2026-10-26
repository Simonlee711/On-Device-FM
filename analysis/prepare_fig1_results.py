#!/usr/bin/env python3
"""Prepare the long-form prediction CSV consumed by plot/fig1.py.

This is an adapter, not an evaluator: it does not run a model or recompute scores.
It standardizes an existing prediction export such as pvc_predictions.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Existing prediction CSV.")
    p.add_argument("--output", type=Path, default=Path("results/fig1_predictions.csv"))
    p.add_argument("--task", default="PVC")
    p.add_argument("--model", default="HiMAE")
    p.add_argument("--label-column", default="label")
    p.add_argument("--score-column", default="p_pvc")
    p.add_argument("--person-column", default="patient_id")
    p.add_argument("--sample-column", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {args.label_column, args.score_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.input}: {sorted(missing)}")

    out = pd.DataFrame({
        "task": args.task,
        "model": args.model,
        "y_true": pd.to_numeric(df[args.label_column], errors="raise").astype(int),
        "y_score": pd.to_numeric(df[args.score_column], errors="raise").astype(float),
    })
    if args.person_column and args.person_column in df.columns:
        out["person_id"] = df[args.person_column]
    if args.sample_column and args.sample_column in df.columns:
        out["sample_id"] = df[args.sample_column]
    else:
        out["sample_id"] = range(len(out))

    if not set(out["y_true"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
