#!/usr/bin/env python3
"""Select model-specific score thresholds on validation predictions only.

Input: canonical predictions (analysis/prediction_schema.py). Only rows with
split == --validation-split are read; test rows cannot influence the thresholds.

For each task/model/regime and target specificity, the selected threshold
minimizes |validation specificity - target| over the distinct validation scores
(a window is called positive when y_score >= threshold). Ties are resolved by
higher validation sensitivity, then by the larger threshold.

Output: task,model,regime,target_specificity,threshold,validation_specificity,
        validation_sensitivity,n_validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from prediction_schema import load_predictions


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--targets", nargs="+", type=float, default=[0.95, 0.90, 0.80])
    p.add_argument("--validation-split", default="val")
    p.add_argument("--output", type=Path, default=Path("results/validation_thresholds.csv"))
    return p.parse_args(argv)


def select_thresholds(pred: pd.DataFrame, targets, validation_split: str = "val") -> pd.DataFrame:
    for target in targets:
        if not (0 < target < 1):
            raise ValueError(f"Target specificity must lie in (0,1): {target}")
    val = pred[pred["split"] == validation_split.lower()]
    if val.empty:
        raise ValueError(f"No rows found for validation split '{validation_split}'.")

    rows = []
    for keys, g in val.groupby(["task", "model", "regime"], sort=False):
        y = g["y_true"].to_numpy(dtype=int)
        s = g["y_score"].to_numpy(dtype=float)
        if np.unique(y).size != 2:
            raise ValueError(f"Validation group {keys} does not contain both classes.")
        fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
        # Drop roc_curve's +inf sentinel ("never predict positive"); it is not a score threshold.
        finite = np.isfinite(thresholds)
        specificity, sensitivity, thresholds = 1.0 - fpr[finite], tpr[finite], thresholds[finite]
        for target in targets:
            dist = np.abs(specificity - target)
            candidates = np.flatnonzero(np.isclose(dist, dist.min(), atol=1e-12, rtol=0))
            best_sens = sensitivity[candidates].max()
            candidates = candidates[np.isclose(sensitivity[candidates], best_sens, atol=1e-12, rtol=0)]
            idx = candidates[np.argmax(thresholds[candidates])]
            rows.append({
                "task": keys[0], "model": keys[1], "regime": keys[2],
                "target_specificity": float(target),
                "threshold": float(thresholds[idx]),
                "validation_specificity": float(specificity[idx]),
                "validation_sensitivity": float(sensitivity[idx]),
                "n_validation": int(len(g)),
            })
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    out = select_thresholds(load_predictions(args.predictions), args.targets, args.validation_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} thresholds to {args.output}")


if __name__ == "__main__":
    main()
