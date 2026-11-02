#!/usr/bin/env python3
"""Prepare the clinical CSV consumed by plot/fig2_3.py from operating points.

By default Figure 3 uses the fine-tuned regime at the validation-selected 95%
specificity target. Change --regime/--target-specificity if the figure definition
changes. This is a formatting/filtering adapter, not a model evaluator.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="operating_points.csv")
    p.add_argument("--regime", default="Fine-tuned")
    p.add_argument("--target-specificity", type=float, default=0.95)
    p.add_argument("--output", type=Path, default=Path("results/fig3_clinical.csv"))
    return p.parse_args()


def norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {"task", "model", "regime", "target_specificity", "tp", "sensitivity", "balanced_accuracy", "youden_index"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    mask = df["regime"].map(norm).eq(norm(args.regime)) & np.isclose(pd.to_numeric(df["target_specificity"]), args.target_specificity)
    out = df.loc[mask, ["task", "model", "tp", "sensitivity", "balanced_accuracy", "youden_index"]].copy()
    if out.empty:
        raise ValueError("No rows matched the requested regime and target specificity.")
    if out.duplicated(["task", "model"]).any():
        raise ValueError("Filtered output contains duplicate task/model rows.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
