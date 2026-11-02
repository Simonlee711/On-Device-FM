"""Shared helpers for participant-level paired bootstrap analyses.

Protocol (Methods, Statistical analysis):
  * point estimates are computed on the original held-out test set;
  * participants are resampled with replacement and all windows of a sampled
    participant are kept together;
  * B = 1,000 replicates, 95% CI = 2.5th / 97.5th percentiles;
  * paired differences are computed within each replicate, using the same
    resampled participants (with the same multiplicities) for both models; the
    two-sided P value is 2 * min{Pr(D <= 0), Pr(D >= 0)} with the finite-replicate
    correction (k + 1) / (B + 1), capped at 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prediction_schema import PAIR_KEYS, normalize_columns  # noqa: F401  (re-exported)


def require(df: pd.DataFrame, cols: set[str], name: str) -> None:
    missing = cols.difference(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def wide_by_model(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot one task/regime/split group to one row per window, one score column per model.

    Pairing is by (person_id, sample_id). The group is rejected, rather than
    silently repaired, when
      * a pairing identifier, label or score is missing;
      * a model has more than one row for the same window;
      * models disagree on y_true for the same window (checked on the raw rows,
        before any deduplication);
      * models were not scored on exactly the same set of windows.
    """
    require(g, {"model", "y_true", "y_score", *PAIR_KEYS}, "Paired predictions")
    cols = ["model", *PAIR_KEYS, "y_true", "y_score"]
    na = [c for c in cols if g[c].isna().any()]
    if na:
        raise ValueError(f"Missing values in {na}; every row needs a pairing identifier, label and score.")

    dup = g.duplicated(["model", *PAIR_KEYS], keep=False)
    if dup.any():
        raise ValueError("Duplicated (model, person_id, sample_id) rows; sample_id must identify one window "
                         "within a participant:\n" + g.loc[dup, ["model", *PAIR_KEYS]].head(10).to_string(index=False))

    n_labels = g.groupby(PAIR_KEYS, sort=False)["y_true"].nunique()
    if (n_labels > 1).any():
        bad = n_labels[n_labels > 1].index[:10].tolist()
        raise ValueError(f"Models disagree on y_true for {int((n_labels > 1).sum())} window(s), e.g. {bad}.")

    models = list(dict.fromkeys(g["model"]))
    scores = g.pivot(index=PAIR_KEYS, columns="model", values="y_score")
    absent = {m: int(scores[m].isna().sum()) for m in models if scores[m].isna().any()}
    if absent:
        raise ValueError(f"Models are not scored on the same windows (windows missing per model: {absent}); "
                         "paired analysis needs a shared window set.")

    labels = g.groupby(PAIR_KEYS, sort=False)["y_true"].first()
    w = scores[models].join(labels).reset_index()
    return w[[*PAIR_KEYS, "y_true", *models]], models


def bootstrap_indices(person: np.ndarray, n_boot: int, seed: int):
    """Yield row-index arrays, one per replicate, resampling participants with replacement.

    The same index array is applied to every model's scores, so both members of a
    paired comparison see identical participants with identical multiplicities.
    """
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(person)
    rows_by_person = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    n = len(uniq)
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        yield np.concatenate([rows_by_person[i] for i in pick])


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_p_value(diffs: np.ndarray) -> float:
    d = diffs[np.isfinite(diffs)]
    b = d.size
    if b == 0:
        return float("nan")
    p_le = (np.sum(d <= 0) + 1) / (b + 1)
    p_ge = (np.sum(d >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(p_le, p_ge)))
