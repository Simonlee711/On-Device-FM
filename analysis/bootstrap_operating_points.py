#!/usr/bin/env python3
"""Participant-bootstrap CIs at fixed, validation-selected operating points (Fig. 3; Extended Data Tables 3-8).

Inputs:
    --predictions  canonical predictions (analysis/prediction_schema.py)
    --thresholds   output of select_validation_thresholds.py

Invariants (tested in tests/test_statistics.py):
  * thresholds come only from the validation split and are applied once to the
    test scores before resampling; they are never re-selected on the test set or
    inside a replicate, so CIs are conditional on the validation-selected
    thresholds;
  * within a task/regime all models are scored on the same windows, and every
    replicate applies one participant resample (same participants, same
    multiplicities) to all models, so dTP = TP_reference - TP_comparator is paired.

For every task/regime/target the test set is resampled at the participant level
(B = 1,000 by default). Within each replicate TP/FP/TN/FN, sensitivity,
specificity, PPV, NPV, balanced accuracy and Youden index are recomputed for every
model, together with dTP. CIs are 2.5th/97.5th percentiles; the dTP P value is
2 * min{Pr(d <= 0), Pr(d >= 0)} with the (k + 1) / (B + 1) correction.

Outputs:
    <output-dir>/operating_points_bootstrap.csv
        one row per task/regime/target/model: point estimates and <metric>_ci_low/_ci_high
    <output-dir>/operating_points_paired_tp.csv
        task,regime,target_specificity,reference,comparator,tp_reference,tp_comparator,
        delta_tp,ci_low,ci_high,p_value,n_boot,seed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_indices, paired_p_value, percentile_ci, wide_by_model
from apply_operating_points import METRICS, confusion_metrics, load_thresholds
from prediction_schema import load_predictions


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--reference-model", default="HiMAE (1.2M)")
    p.add_argument("--test-split", default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args(argv)


def bootstrap_operating_points(pred: pd.DataFrame, thr: pd.DataFrame, reference: str, n_boot: int,
                               seed: int, test_split: str = "test", return_replicates: bool = False):
    test = pred[pred["split"] == test_split.lower()]
    if test.empty:
        raise ValueError(f"No rows found for test split '{test_split}'.")
    tested = set(map(tuple, test[["task", "model", "regime"]].drop_duplicates().to_numpy()))
    orphan = [k for k in map(tuple, thr[["task", "model", "regime"]].drop_duplicates().to_numpy()) if k not in tested]
    if orphan:
        raise ValueError(f"Thresholds without test predictions: {orphan}")

    rows, paired, replicates = [], [], {}
    for (task, regime), g in test.groupby(["task", "regime"], sort=False):
        w, models = wide_by_model(g)
        y = w["y_true"].to_numpy(dtype=int)
        person = w["person_id"].to_numpy()
        n_part = int(w["person_id"].nunique())
        t_grp = thr[(thr.task == task) & (thr.regime == regime)]
        for target, tg in t_grp.groupby("target_specificity", sort=False):
            cut = dict(zip(tg["model"], tg["threshold"]))
            have = [m for m in models if m in cut]
            if len(have) < len(models):
                print(f"[warn] no threshold for {sorted(set(models) - set(have))} in {task}/{regime}/{target}.")
            # Thresholds are applied once here; replicates only re-index these fixed predictions.
            yhat = {m: (w[m].to_numpy(dtype=float) >= cut[m]).astype(int) for m in have}
            point = {m: confusion_metrics(y, yhat[m]) for m in have}

            boot = {m: {k: np.empty(n_boot) for k in METRICS} for m in have}
            for b, idx in enumerate(bootstrap_indices(person, n_boot, seed)):
                yb = y[idx]
                for m in have:
                    for k, v in confusion_metrics(yb, yhat[m][idx]).items():
                        boot[m][k][b] = v
            replicates[(task, regime, float(target))] = boot

            for m in have:
                r = {"task": task, "regime": regime, "target_specificity": float(target),
                     "model": m, "threshold": cut[m]}
                for k in METRICS:
                    r[k] = point[m][k]
                    r[f"{k}_ci_low"], r[f"{k}_ci_high"] = percentile_ci(boot[m][k])
                r.update({"n_test": len(w), "n_participants": n_part, "n_boot": n_boot, "seed": seed})
                rows.append(r)

            if reference not in have:
                print(f"[warn] reference model '{reference}' absent for {task}/{regime}/{target}; skipping paired TP.")
                continue
            for m in have:
                if m == reference:
                    continue
                d = boot[reference]["tp"] - boot[m]["tp"]
                lo, hi = percentile_ci(d)
                paired.append({
                    "task": task, "regime": regime, "target_specificity": float(target),
                    "reference": reference, "comparator": m,
                    "tp_reference": point[reference]["tp"], "tp_comparator": point[m]["tp"],
                    "delta_tp": point[reference]["tp"] - point[m]["tp"],
                    "ci_low": lo, "ci_high": hi, "p_value": paired_p_value(d),
                    "n_boot": n_boot, "seed": seed,
                })

    out = pd.DataFrame(rows), pd.DataFrame(paired)
    return (*out, replicates) if return_replicates else out


def main(argv=None):
    args = parse_args(argv)
    rows, paired = bootstrap_operating_points(load_predictions(args.predictions), load_thresholds(args.thresholds),
                                              args.reference_model, args.n_boot, args.seed, args.test_split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "operating_points_bootstrap.csv", index=False)
    paired.to_csv(args.output_dir / "operating_points_paired_tp.csv", index=False)
    print(f"Wrote {len(rows)} operating-point rows and {len(paired)} paired TP rows to {args.output_dir}")


if __name__ == "__main__":
    main()
