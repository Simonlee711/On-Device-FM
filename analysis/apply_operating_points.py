#!/usr/bin/env python3
"""Apply validation-selected thresholds to held-out predictions.

This script computes deterministic point-estimate confusion counts and derived
metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PRED_REQ = {"task", "model", "regime", "split", "y_true", "y_score"}
THR_REQ = {"task", "model", "regime", "target_specificity", "threshold"}


def div(a, b):
    return float(a / b) if b else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--test-split", default="test")
    p.add_argument("--output", type=Path, default=Path("results/operating_points.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    thr = pd.read_csv(args.thresholds)
    pred.columns = [str(c).strip().lower() for c in pred.columns]
    thr.columns = [str(c).strip().lower() for c in thr.columns]
    if PRED_REQ.difference(pred.columns):
        raise ValueError(f"Predictions missing: {sorted(PRED_REQ.difference(pred.columns))}")
    if THR_REQ.difference(thr.columns):
        raise ValueError(f"Thresholds missing: {sorted(THR_REQ.difference(thr.columns))}")

    pred["split"] = pred["split"].astype(str).str.lower().str.strip()
    test = pred[pred["split"] == args.test_split.lower()].copy()
    rows = []
    for t in thr.itertuples(index=False):
        g = test[(test.task == t.task) & (test.model == t.model) & (test.regime == t.regime)]
        if g.empty:
            raise ValueError(f"No test predictions for {(t.task, t.model, t.regime)}")
        y = pd.to_numeric(g.y_true, errors="raise").to_numpy(dtype=int)
        s = pd.to_numeric(g.y_score, errors="raise").to_numpy(dtype=float)
        yh = (s >= float(t.threshold)).astype(int)
        tp = int(np.sum((y == 1) & (yh == 1)))
        fp = int(np.sum((y == 0) & (yh == 1)))
        tn = int(np.sum((y == 0) & (yh == 0)))
        fn = int(np.sum((y == 1) & (yh == 0)))
        sens = div(tp, tp + fn); spec = div(tn, tn + fp)
        ppv = div(tp, tp + fp); npv = div(tn, tn + fn)
        bal = (sens + spec) / 2.0
        rows.append({
            "task": t.task, "model": t.model, "regime": t.regime,
            "target_specificity": float(t.target_specificity), "threshold": float(t.threshold),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "balanced_accuracy": bal, "youden_index": sens + spec - 1.0,
            "n_test": int(len(g)),
        })
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} operating points to {args.output}")


if __name__ == "__main__":
    main()
