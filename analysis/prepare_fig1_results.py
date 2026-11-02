#!/usr/bin/env python3
"""Select the held-out rows of a canonical prediction table for plot/fig1.py (Fig. 1, panel 3 ROC curves).

This is an adapter, not an evaluator: it does not run a model or recompute scores.
Input is a canonical prediction table (analysis/prediction_schema.py); legacy
exports must first be converted with analysis/convert_predictions.py.

Fig. 1 panel 3 shows linear probes on frozen representations, so --regime
defaults to "Frozen". Output columns: task,model,person_id,sample_id,y_true,y_score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from prediction_schema import load_predictions


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Canonical prediction CSV.")
    p.add_argument("--output", type=Path, default=Path("results/fig1_predictions.csv"))
    p.add_argument("--split", default="test")
    p.add_argument("--regime", default="Frozen", help='Regime to keep; pass "" to keep all regimes.')
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = load_predictions(args.input)
    df = df[df["split"] == args.split.lower()]
    if args.regime:
        df = df[df["regime"].astype(str).str.lower() == args.regime.lower()]
    if df.empty:
        raise ValueError(f"No rows with split == '{args.split}' and regime == '{args.regime}'.")
    if df.groupby(["task", "model"])["regime"].nunique().gt(1).any():
        raise ValueError("Several regimes per task/model; pass --regime to select one.")
    out = df[["task", "model", "person_id", "sample_id", "y_true", "y_score"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
