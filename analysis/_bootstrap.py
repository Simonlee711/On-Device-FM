"""Shared helpers for participant-level paired bootstrap analyses.

Protocol (Methods, Statistical analysis):
  * point estimates are computed on the original held-out test set;
  * participants are resampled with replacement and all windows of a sampled
    participant are kept together;
  * B = 1,000 replicates, 95% CI = 2.5th / 97.5th percentiles;
  * paired differences are computed within each replicate, and the two-sided P
    value is 2 * min{Pr(D <= 0), Pr(D >= 0)} with the finite-replicate correction
    (k + 1) / (B + 1), capped at 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def require(df: pd.DataFrame, cols: set[str], name: str) -> None:
    missing = cols.difference(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def wide_by_model(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot one task/regime/split group to one row per window, one score column per model.

    Models must be scored on the same windows (sample_id within person_id) so that
    bootstrap replicates are paired across models.
    """
    if "sample_id" not in g.columns:
        g = g.assign(sample_id=g.groupby("model").cumcount())
    keys = ["person_id", "sample_id"]
    if g.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate (person_id, sample_id, model) rows; sample_id must be unique per window.")
    labels = g.drop_duplicates(keys)[keys + ["y_true"]]
    if labels.duplicated(keys).any():
        raise ValueError("Models disagree on y_true for the same window.")
    scores = g.pivot(index=keys, columns="model", values="y_score")
    models = list(dict.fromkeys(g["model"]))
    w = labels.set_index(keys).join(scores, how="inner").reset_index()
    if w[models].isna().any().any():
        bad = [m for m in models if w[m].isna().any()]
        raise ValueError(f"Models {bad} are missing scores for some windows; paired analysis needs a shared window set.")
    return w, models


def bootstrap_indices(person: np.ndarray, n_boot: int, seed: int):
    """Yield row-index arrays, one per replicate, resampling participants with replacement."""
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
