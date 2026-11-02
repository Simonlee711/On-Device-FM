#!/usr/bin/env python3
"""Apply validation-selected thresholds once to the held-out test predictions.

Inputs:
    --predictions  canonical predictions (analysis/prediction_schema.py)
    --thresholds   output of select_validation_thresholds.py
                   (task,model,regime,target_specificity,threshold,...)

A test window is called positive when y_score >= threshold. Thresholds are used
exactly as supplied; nothing is re-estimated on the test split. Output is one row
per task/model/regime/target with deterministic point estimates:
    tp,fp,tn,fn,sensitivity,specificity,ppv,npv,balanced_accuracy,youden_index,n_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_schema import load_predictions, normalize_columns

THR_KEYS = ["task", "model", "regime", "target_specificity"]
COUNTS = ["tp", "fp", "tn", "fn"]
METRICS = COUNTS + ["sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "youden_index"]


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def confusion_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y == 1) & (yhat == 1)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens, "specificity": spec,
            "ppv": _div(tp, tp + fp), "npv": _div(tn, tn + fn),
            "balanced_accuracy": (sens + spec) / 2.0, "youden_index": sens + spec - 1.0}


def load_thresholds(path_or_df) -> pd.DataFrame:
    thr = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    thr = normalize_columns(thr)
    missing = [c for c in THR_KEYS + ["threshold"] if c not in thr.columns]
    if missing:
        raise ValueError(f"Thresholds missing columns: {missing}")
    thr["target_specificity"] = thr["target_specificity"].astype(float)
    thr["threshold"] = thr["threshold"].astype(float)
    if thr[THR_KEYS + ["threshold"]].isna().any().any():
        raise ValueError("Thresholds contain missing values.")
    if thr.duplicated(THR_KEYS).any():
        raise ValueError("Thresholds must have one row per task/model/regime/target_specificity.")
    return thr


def apply_thresholds(pred: pd.DataFrame, thr: pd.DataFrame, test_split: str = "test") -> pd.DataFrame:
    test = pred[pred["split"] == test_split.lower()]
    rows = []
    for t in thr.itertuples(index=False):
        g = test[(test.task == t.task) & (test.model == t.model) & (test.regime == t.regime)]
        if g.empty:
            raise ValueError(f"No test predictions for {(t.task, t.model, t.regime)}")
        y = g["y_true"].to_numpy(dtype=int)
        yhat = (g["y_score"].to_numpy(dtype=float) >= t.threshold).astype(int)
        rows.append({"task": t.task, "model": t.model, "regime": t.regime,
                     "target_specificity": t.target_specificity, "threshold": t.threshold,
                     **confusion_metrics(y, yhat), "n_test": int(len(g))})
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--test-split", default="test")
    p.add_argument("--output", type=Path, default=Path("results/operating_points.csv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = apply_thresholds(load_predictions(args.predictions), load_thresholds(args.thresholds), args.test_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} operating points to {args.output}")


if __name__ == "__main__":
    main()
