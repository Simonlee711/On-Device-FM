#!/usr/bin/env python3
"""Participant-level bootstrap AUROC with paired differences against a reference model.

Input is a long-form prediction export with columns:
    task,model,regime,split,person_id,y_true,y_score
Optional: sample_id (window identifier within participant; required when models
are exported in different row orders).

For each task/regime on the test split, AUROC is computed for every model on the
original test set (point estimate) and within 1,000 participant-level bootstrap
replicates (95% percentile CI). Within each replicate the paired difference
    dAUROC = AUROC_reference - AUROC_comparator
is recorded, and a two-sided bootstrap P value is computed as
2 * min{Pr(d <= 0), Pr(d >= 0)} with the (k + 1) / (B + 1) correction.
Replicates containing a single class are skipped and counted in n_valid_boot.

Outputs:
    <output-dir>/auroc_summary.csv          task,regime,model,auroc,ci_low,ci_high,...
    <output-dir>/auroc_paired_differences.csv
        task,regime,reference,comparator,delta_auroc,ci_low,ci_high,p_value,...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _bootstrap import (bootstrap_indices, normalize_columns, paired_p_value,
                        percentile_ci, require, wide_by_model)

REQ = {"task", "model", "regime", "split", "person_id", "y_true", "y_score"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference-model", default="HiMAE (1.2M)")
    p.add_argument("--test-split", default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if np.unique(y).size == 2 else float("nan")


def main():
    args = parse_args()
    df = normalize_columns(pd.read_csv(args.predictions))
    require(df, REQ, "Predictions")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    test = df[df["split"] == args.test_split.lower()]
    if test.empty:
        raise ValueError(f"No rows found for test split '{args.test_split}'.")

    summary, paired = [], []
    for (task, regime), g in test.groupby(["task", "regime"], sort=False):
        w, models = wide_by_model(g)
        y = w["y_true"].to_numpy(dtype=int)
        s = {m: w[m].to_numpy(dtype=float) for m in models}
        point = {m: safe_auc(y, s[m]) for m in models}

        boot = {m: np.empty(args.n_boot) for m in models}
        for b, idx in enumerate(bootstrap_indices(w["person_id"].to_numpy(), args.n_boot, args.seed)):
            yb = y[idx]
            for m in models:
                boot[m][b] = safe_auc(yb, s[m][idx])

        n_part = w["person_id"].nunique()
        for m in models:
            lo, hi = percentile_ci(boot[m])
            summary.append({
                "task": task, "regime": regime, "model": m, "auroc": point[m],
                "ci_low": lo, "ci_high": hi,
                "n_windows": len(w), "n_positive": int(y.sum()), "n_participants": n_part,
                "n_boot": args.n_boot, "n_valid_boot": int(np.isfinite(boot[m]).sum()), "seed": args.seed,
            })

        if args.reference_model not in models:
            print(f"[warn] reference model '{args.reference_model}' absent for {task}/{regime}; skipping paired tests.")
            continue
        ref = args.reference_model
        for m in models:
            if m == ref:
                continue
            d = boot[ref] - boot[m]
            lo, hi = percentile_ci(d)
            paired.append({
                "task": task, "regime": regime, "reference": ref, "comparator": m,
                "delta_auroc": point[ref] - point[m], "ci_low": lo, "ci_high": hi,
                "p_value": paired_p_value(d), "n_valid_boot": int(np.isfinite(d).sum()),
                "n_boot": args.n_boot, "seed": args.seed,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output_dir / "auroc_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output_dir / "auroc_paired_differences.csv", index=False)
    print(f"Wrote {len(summary)} AUROC rows and {len(paired)} paired differences to {args.output_dir}")


if __name__ == "__main__":
    main()
