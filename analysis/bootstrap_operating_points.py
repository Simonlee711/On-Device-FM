#!/usr/bin/env python3
"""Bootstrap confidence intervals at fixed, validation-selected operating points.

Inputs:
    --predictions  long-form export: task,model,regime,split,person_id,y_true,y_score
                   (optional sample_id)
    --thresholds   output of select_validation_thresholds.py:
                   task,model,regime,target_specificity,threshold

Thresholds are selected once on the validation split and are held fixed here; they
are never re-optimized on the test set or inside bootstrap replicates. For every
task/regime/target specificity, the test set is resampled at the participant level
(B = 1,000 by default). Within each replicate we recompute TP/FP/TN/FN,
sensitivity, specificity, PPV, NPV, balanced accuracy and Youden index for every
model, and the paired true-positive difference
    dTP = TP_reference - TP_comparator.
CIs are 2.5th/97.5th percentiles; P values are 2 * min{Pr(d <= 0), Pr(d >= 0)}
with the (k + 1) / (B + 1) correction.

Outputs:
    <output-dir>/operating_points_bootstrap.csv
        one row per task/regime/target/model with point estimates and *_ci_low/_ci_high
    <output-dir>/operating_points_paired_tp.csv
        task,regime,target_specificity,reference,comparator,delta_tp,ci_low,ci_high,p_value
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import (bootstrap_indices, normalize_columns, paired_p_value,
                        percentile_ci, require, wide_by_model)

PRED_REQ = {"task", "model", "regime", "split", "person_id", "y_true", "y_score"}
THR_REQ = {"task", "model", "regime", "target_specificity", "threshold"}
METRICS = ["tp", "fp", "tn", "fn", "sensitivity", "specificity", "ppv", "npv",
           "balanced_accuracy", "youden_index"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--reference-model", default="HiMAE (1.2M)")
    p.add_argument("--test-split", default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def confusion_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((y == 1) & (yhat == 1)))
    fp = float(np.sum((y == 0) & (yhat == 1)))
    tn = float(np.sum((y == 0) & (yhat == 0)))
    fn = float(np.sum((y == 1) & (yhat == 0)))
    sens = float(_div(tp, tp + fn)); spec = float(_div(tn, tn + fp))
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        "ppv": float(_div(tp, tp + fp)), "npv": float(_div(tn, tn + fn)),
        "balanced_accuracy": (sens + spec) / 2.0, "youden_index": sens + spec - 1.0,
    }


def main():
    args = parse_args()
    pred = normalize_columns(pd.read_csv(args.predictions))
    thr = normalize_columns(pd.read_csv(args.thresholds))
    require(pred, PRED_REQ, "Predictions")
    require(thr, THR_REQ, "Thresholds")
    pred["split"] = pred["split"].astype(str).str.lower().str.strip()
    test = pred[pred["split"] == args.test_split.lower()]
    if test.empty:
        raise ValueError(f"No rows found for test split '{args.test_split}'.")

    rows, paired = [], []
    for (task, regime), g in test.groupby(["task", "regime"], sort=False):
        w, models = wide_by_model(g)
        y = w["y_true"].to_numpy(dtype=int)
        person = w["person_id"].to_numpy()
        t_grp = thr[(thr.task == task) & (thr.regime == regime)]
        for target, tg in t_grp.groupby("target_specificity", sort=False):
            cut = dict(zip(tg["model"], tg["threshold"].astype(float)))
            have = [m for m in models if m in cut]
            if not have:
                continue
            yhat = {m: (w[m].to_numpy(dtype=float) >= cut[m]).astype(int) for m in have}
            point = {m: confusion_metrics(y, yhat[m]) for m in have}

            boot = {m: {k: np.empty(args.n_boot) for k in METRICS} for m in have}
            for b, idx in enumerate(bootstrap_indices(person, args.n_boot, args.seed)):
                yb = y[idx]
                for m in have:
                    for k, v in confusion_metrics(yb, yhat[m][idx]).items():
                        boot[m][k][b] = v

            for m in have:
                r = {"task": task, "regime": regime, "target_specificity": float(target),
                     "model": m, "threshold": cut[m]}
                for k in METRICS:
                    r[k] = int(point[m][k]) if k in ("tp", "fp", "tn", "fn") else point[m][k]
                    r[f"{k}_ci_low"], r[f"{k}_ci_high"] = percentile_ci(boot[m][k])
                r.update({"n_test": len(w), "n_participants": int(pd.Series(person).nunique()),
                          "n_boot": args.n_boot, "seed": args.seed})
                rows.append(r)

            ref = args.reference_model
            if ref not in have:
                print(f"[warn] reference model '{ref}' absent for {task}/{regime}/{target}; skipping paired TP.")
                continue
            for m in have:
                if m == ref:
                    continue
                d = boot[ref]["tp"] - boot[m]["tp"]
                lo, hi = percentile_ci(d)
                paired.append({
                    "task": task, "regime": regime, "target_specificity": float(target),
                    "reference": ref, "comparator": m,
                    "tp_reference": int(point[ref]["tp"]), "tp_comparator": int(point[m]["tp"]),
                    "delta_tp": int(point[ref]["tp"] - point[m]["tp"]),
                    "ci_low": lo, "ci_high": hi, "p_value": paired_p_value(d),
                    "n_boot": args.n_boot, "seed": args.seed,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "operating_points_bootstrap.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output_dir / "operating_points_paired_tp.csv", index=False)
    print(f"Wrote {len(rows)} operating-point rows and {len(paired)} paired TP rows to {args.output_dir}")


if __name__ == "__main__":
    main()
