#!/usr/bin/env python3
"""Select model-specific score thresholds on validation predictions only.

Input is a long-form prediction export with columns:
    task,model,regime,split,y_true,y_score
Optional: person_id,sample_id

For each task/model/regime and target specificity, the selected threshold minimizes
absolute validation-specificity error. Ties are resolved by higher validation
sensitivity, then by the larger threshold for deterministic output.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

REQ = {"task", "model", "regime", "split", "y_true", "y_score"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--targets", nargs="+", type=float, default=[0.95, 0.90, 0.80])
    p.add_argument("--validation-split", default="val")
    p.add_argument("--output", type=Path, default=Path("results/validation_thresholds.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = REQ.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    val = df[df["split"] == args.validation_split.lower()].copy()
    if val.empty:
        raise ValueError(f"No rows found for validation split '{args.validation_split}'.")

    rows = []
    for keys, g in val.groupby(["task", "model", "regime"], sort=False):
        y = pd.to_numeric(g["y_true"], errors="raise").to_numpy(dtype=int)
        s = pd.to_numeric(g["y_score"], errors="raise").to_numpy(dtype=float)
        if np.unique(y).size != 2:
            raise ValueError(f"Validation group {keys} does not contain both classes.")
        fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
        # Drop roc_curve's +inf sentinel ("never predict positive"); it is not a score threshold.
        finite = np.isfinite(thresholds)
        fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
        specificity = 1.0 - fpr
        sensitivity = tpr
        for target in args.targets:
            if not (0 < target < 1):
                raise ValueError(f"Target specificity must lie in (0,1): {target}")
            dist = np.abs(specificity - target)
            best_dist = dist.min()
            candidates = np.flatnonzero(np.isclose(dist, best_dist, atol=1e-12, rtol=0))
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

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} thresholds to {args.output}")


if __name__ == "__main__":
    main()
