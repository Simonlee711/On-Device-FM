#!/usr/bin/env python3
"""AUROC point estimates, participant-bootstrap CIs and paired DeLong tests (Table 1, Results 2.1).

Input: canonical predictions (analysis/prediction_schema.py)
    task,model,regime,split,person_id,sample_id,y_true,y_score
Only --test-split rows are used. Within each task/regime, every model must be
scored on the same (person_id, sample_id) windows.

What the manuscript reports, and which column holds it
------------------------------------------------------
* Table 1 AUROC and its 95% CI
      auroc_summary.csv: auroc, participant_bootstrap_ci_low, participant_bootstrap_ci_high
  Percentile CI over B participant-level bootstrap replicates (participants
  resampled with replacement, all windows of a participant kept together).
* Results 2.1, dAUROC = AUROC_reference - AUROC_comparator, its 95% CI and P value
      auroc_paired_differences.csv: delta_auroc,
          participant_bootstrap_ci_low, participant_bootstrap_ci_high   (the 95% CI)
          paired_delong_p_value                                         (the P value)
  The CI is the percentile interval of the paired bootstrap differences (same
  resampled participants, same multiplicities, for both models). The P value is
  the ordinary two-sided paired DeLong test (DeLong, DeLong & Clarke-Pearson 1988;
  Sun & Xu 2014 fast algorithm) on the full test set, which treats windows as
  independent observations.

Additional columns (not reported in the manuscript; sensitivity analyses)
-------------------------------------------------------------------------
* paired_delong_se / _z / _wald_ci_low / _wald_ci_high, and delong_* in the
  summary: Wald quantities from the same ordinary DeLong covariance.
* clustered_delong_*: participant-clustered DeLong (Obuchowski 1997), which sums
  the DeLong structural components within participant so that within-participant
  correlation enters the variance.
* participant_bootstrap_p_value: 2 * min{Pr(d <= 0), Pr(d >= 0)} over the paired
  bootstrap replicates with the (k + 1) / (B + 1) correction (smallest attainable
  value 2 / (B + 1), i.e. 0.002 for B = 1,000).

Bootstrap replicates in which the resampled test set has a single class give an
undefined AUROC; they are excluded and counted in n_valid_boot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _bootstrap import bootstrap_indices, paired_p_value, percentile_ci, wide_by_model
from delong import delong_auc_ci, delong_paired_test
from prediction_schema import load_predictions

MANUSCRIPT_P_VALUE = "paired_delong_p_value"
MANUSCRIPT_CI = ("participant_bootstrap_ci_low", "participant_bootstrap_ci_high")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference-model", default="HiMAE (1.2M)")
    p.add_argument("--test-split", default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args(argv)


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if np.unique(y).size == 2 else float("nan")


def _nan_if_fails(fn, keys):
    try:
        return fn()
    except ValueError:
        return {k: float("nan") for k in keys}


def delong_single(y, score, groups, prefix):
    keys = [f"{prefix}_se", f"{prefix}_wald_ci_low", f"{prefix}_wald_ci_high"]

    def run():
        _, se, lo, hi = delong_auc_ci(y, score, groups)
        return dict(zip(keys, (se, lo, hi)))
    return _nan_if_fails(run, keys)


def delong_pair(y, a, b, groups, prefix):
    src = {"se": "se", "z": "z", "p_value": "p_value", "wald_ci_low": "ci_low", "wald_ci_high": "ci_high"}

    def run():
        r = delong_paired_test(y, a, b, groups)
        return {f"{prefix}_{k}": r[v] for k, v in src.items()}
    return _nan_if_fails(run, [f"{prefix}_{k}" for k in src])


def bootstrap_aucs(y: np.ndarray, scores: dict[str, np.ndarray], person: np.ndarray,
                   n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """AUROC of every model in each participant-level replicate (shared indices across models)."""
    boot = {m: np.empty(n_boot) for m in scores}
    for b, idx in enumerate(bootstrap_indices(person, n_boot, seed)):
        yb = y[idx]
        for m, s in scores.items():
            boot[m][b] = safe_auc(yb, s[idx])
    return boot


def paired_auroc_stats(y, score_a, score_b, person, boot_a, boot_b) -> dict[str, float]:
    """delta = AUROC_a - AUROC_b with every inference method, named by method (see module docstring)."""
    d = boot_a - boot_b
    lo, hi = percentile_ci(d)
    return {
        "delta_auroc": safe_auc(y, score_a) - safe_auc(y, score_b),
        "participant_bootstrap_ci_low": lo, "participant_bootstrap_ci_high": hi,
        **delong_pair(y, score_a, score_b, None, "paired_delong"),
        **delong_pair(y, score_a, score_b, person, "clustered_delong"),
        "participant_bootstrap_p_value": paired_p_value(d),
        "n_valid_boot": int(np.isfinite(d).sum()),
    }


def compare_aurocs(pred: pd.DataFrame, reference: str, n_boot: int, seed: int,
                   test_split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pred[pred["split"] == test_split.lower()]
    if test.empty:
        raise ValueError(f"No rows found for test split '{test_split}'.")

    summary, paired = [], []
    for (task, regime), g in test.groupby(["task", "regime"], sort=False):
        w, models = wide_by_model(g)
        y = w["y_true"].to_numpy(dtype=int)
        person = w["person_id"].to_numpy()
        s = {m: w[m].to_numpy(dtype=float) for m in models}
        boot = bootstrap_aucs(y, s, person, n_boot, seed)
        common = {"n_windows": len(w), "n_positive": int(y.sum()), "n_participants": int(w["person_id"].nunique())}

        for m in models:
            lo, hi = percentile_ci(boot[m])
            summary.append({
                "task": task, "regime": regime, "model": m, **common,
                "auroc": safe_auc(y, s[m]),
                "participant_bootstrap_ci_low": lo, "participant_bootstrap_ci_high": hi,
                **delong_single(y, s[m], None, "delong"),
                **delong_single(y, s[m], person, "clustered_delong"),
                "n_boot": n_boot, "n_valid_boot": int(np.isfinite(boot[m]).sum()), "seed": seed,
            })

        if reference not in models:
            print(f"[warn] reference model '{reference}' absent for {task}/{regime}; skipping paired tests.")
            continue
        for m in models:
            if m == reference:
                continue
            paired.append({
                "task": task, "regime": regime, "reference": reference, "comparator": m, **common,
                "auroc_reference": safe_auc(y, s[reference]), "auroc_comparator": safe_auc(y, s[m]),
                **paired_auroc_stats(y, s[reference], s[m], person, boot[reference], boot[m]),
                "n_boot": n_boot, "seed": seed,
            })
    return pd.DataFrame(summary), pd.DataFrame(paired)


def main(argv=None):
    args = parse_args(argv)
    pred = load_predictions(args.predictions)
    summary, paired = compare_aurocs(pred, args.reference_model, args.n_boot, args.seed, args.test_split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "auroc_summary.csv", index=False)
    paired.to_csv(args.output_dir / "auroc_paired_differences.csv", index=False)
    print(f"Wrote {len(summary)} AUROC rows and {len(paired)} paired differences to {args.output_dir}")
    print(f"Manuscript values: AUROC CI = {', '.join(MANUSCRIPT_CI)}; "
          f"dAUROC CI = {', '.join(MANUSCRIPT_CI)}; dAUROC P = {MANUSCRIPT_P_VALUE}.")


if __name__ == "__main__":
    main()
