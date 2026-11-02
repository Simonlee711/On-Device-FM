"""Canonical prediction schema shared by every exporter and statistical script.

One row per (task, model, regime, split, window):

    task,model,regime,split,person_id,sample_id,y_true,y_score

* task      : downstream task name, e.g. "PVC".
* model     : model display name, e.g. "HiMAE (1.2M)".
* regime    : evaluation regime, e.g. "Frozen" or "Fine-tuned".
* split     : "val" (threshold selection only) or "test" (held out).
* person_id : participant identifier; bootstrap replicates resample participants.
* sample_id : window identifier, unique within participant. Together with
              person_id it is the pairing key: paired analyses require every model
              of a task/regime to be scored on exactly the same (person_id, sample_id)
              windows with the same y_true.
* y_true    : binary label (0/1).
* y_score   : continuous score for the positive class (higher = more likely positive).

Exporters (HiMAE_PVC_Detection.ipynb, pvc/downstream_eval/binary_linear_prob.py,
fewshot_finetune.py --predictions-output, multiresolution_probe.py) write this
schema directly. Older files are converted with analysis/convert_predictions.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = ["task", "model", "regime", "split", "person_id", "sample_id", "y_true", "y_score"]
PAIR_KEYS = ["person_id", "sample_id"]
GROUP_KEYS = ["task", "model", "regime", "split"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def validate_predictions(df: pd.DataFrame, name: str = "Predictions",
                         required: list[str] | None = None) -> pd.DataFrame:
    """Return a cleaned copy of a canonical prediction table, or raise ValueError.

    Checks: required columns present; no missing values in them; y_true binary;
    y_score finite; split normalized to lower case; no duplicated window within a
    task/model/regime/split.
    """
    required = CANONICAL_COLUMNS if required is None else required
    df = normalize_columns(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        hint = ""
        if {"patient_id", "label", "p_pvc"} & set(df.columns):
            hint = " This looks like a legacy export; convert it with analysis/convert_predictions.py."
        raise ValueError(f"{name} missing columns: {missing}.{hint}")
    na = [c for c in required if df[c].isna().any()]
    if na:
        raise ValueError(f"{name} has missing values in: {na}")
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
    if "y_true" in df.columns:
        y = pd.to_numeric(df["y_true"], errors="raise")
        if not y.isin([0, 1]).all():
            raise ValueError(f"{name}: y_true must be binary 0/1.")
        df["y_true"] = y.astype(int)
    if "y_score" in df.columns:
        s = pd.to_numeric(df["y_score"], errors="raise").astype(float)
        if not np.isfinite(s).all():
            raise ValueError(f"{name}: y_score must be finite.")
        df["y_score"] = s
    keys = [c for c in GROUP_KEYS + PAIR_KEYS if c in df.columns]
    if set(PAIR_KEYS) <= set(keys):
        dup = df.duplicated(keys, keep=False)
        if dup.any():
            raise ValueError(f"{name}: duplicated window rows for {keys}:\n"
                             + df.loc[dup, keys].head(10).to_string(index=False))
    return df


def load_predictions(path: Path, name: str | None = None) -> pd.DataFrame:
    return validate_predictions(pd.read_csv(path), name or str(path))


def make_predictions_frame(task: str, model: str, regime: str, split, person_id, sample_id,
                           y_true, y_score) -> pd.DataFrame:
    """Build a validated canonical table from arrays (split may be a scalar or an array)."""
    n = len(np.asarray(y_true))
    df = pd.DataFrame({
        "task": task, "model": model, "regime": regime,
        "split": split if np.ndim(split) else [split] * n,
        "person_id": np.asarray(person_id), "sample_id": np.asarray(sample_id),
        "y_true": np.asarray(y_true).astype(int), "y_score": np.asarray(y_score, dtype=float),
    }, columns=CANONICAL_COLUMNS)
    return validate_predictions(df, f"{task}/{model}/{regime}")


def window_ids_within_participant(person_id: np.ndarray) -> np.ndarray:
    """sample_id = position of each window within its participant, in dataset order.

    Exporters call this on the full dataset (before splitting), so a window keeps
    the same sample_id regardless of which model or split exports it.
    """
    return pd.Series(np.arange(len(person_id))).groupby(np.asarray(person_id)).cumcount().to_numpy()
