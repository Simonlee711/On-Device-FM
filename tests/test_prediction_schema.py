"""Canonical prediction schema, legacy conversion and paired-alignment validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from _bootstrap import wide_by_model
from conftest import REPO
from convert_predictions import convert
from prediction_schema import CANONICAL_COLUMNS, make_predictions_frame, validate_predictions


def two_model_group() -> pd.DataFrame:
    base = pd.DataFrame({"person_id": [1, 1, 2, 2], "sample_id": [0, 1, 0, 1], "y_true": [0, 1, 0, 1]})
    a = base.assign(model="A", y_score=[0.1, 0.9, 0.2, 0.8])
    b = base.assign(model="B", y_score=[0.3, 0.6, 0.4, 0.7])
    return pd.concat([a, b], ignore_index=True)


# --- paired alignment (analysis/_bootstrap.py::wide_by_model) ---------------------------

def test_wide_by_model_pairs_by_window():
    w, models = wide_by_model(two_model_group().sample(frac=1, random_state=0))
    assert models == ["A", "B"] or models == ["B", "A"]
    assert len(w) == 4
    row = w[(w.person_id == 2) & (w.sample_id == 1)].iloc[0]
    assert (row.y_true, row.A, row.B) == (1, 0.8, 0.7)


def test_label_disagreement_is_detected_before_deduplication():
    g = two_model_group()
    g.loc[(g.model == "B") & (g.person_id == 1) & (g.sample_id == 0), "y_true"] = 1
    with pytest.raises(ValueError, match="disagree on y_true"):
        wide_by_model(g)


def test_missing_sample_id_column_is_rejected():
    with pytest.raises(ValueError, match="sample_id"):
        wide_by_model(two_model_group().drop(columns="sample_id"))


@pytest.mark.parametrize("col", ["person_id", "sample_id", "y_true", "y_score"])
def test_missing_values_are_rejected(col):
    g = two_model_group()
    g[col] = g[col].astype(float)
    g.loc[0, col] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        wide_by_model(g)


def test_duplicated_window_within_model_is_rejected():
    g = two_model_group()
    g = pd.concat([g, g.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicated"):
        wide_by_model(g)


def test_unshared_window_set_is_rejected():
    g = two_model_group()
    g = g.drop(index=g[(g.model == "B") & (g.person_id == 2) & (g.sample_id == 1)].index)
    with pytest.raises(ValueError, match="same windows"):
        wide_by_model(g)


# --- canonical schema --------------------------------------------------------------------

def test_validate_rejects_legacy_export_with_hint():
    legacy = pd.DataFrame({"patient_id": [1], "label": [0], "p_pvc": [0.2]})
    with pytest.raises(ValueError, match="convert_predictions.py"):
        validate_predictions(legacy)


def test_validate_rejects_nonbinary_labels_and_duplicates():
    df = make_predictions_frame("PVC", "M", "Frozen", "test", [1, 1], [0, 1], [0, 1], [0.1, 0.9])
    bad = df.assign(y_true=[0, 2])
    with pytest.raises(ValueError, match="binary"):
        validate_predictions(bad)
    with pytest.raises(ValueError, match="duplicated"):
        validate_predictions(pd.concat([df, df.iloc[[0]]]))


def test_make_predictions_frame_has_canonical_columns():
    df = make_predictions_frame("PVC", "M", "Frozen", np.array(["val", "test"]), [1, 2], [0, 0], [0, 1], [0.1, 0.9])
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df.split.tolist() == ["val", "test"]


# --- legacy conversion (analysis/convert_predictions.py) ----------------------------------

def test_convert_requires_explicit_split_and_window_ids():
    legacy = pd.DataFrame({"patient_id": [1, 1, 2], "label": [0, 1, 0], "p_pvc": [0.2, 0.8, 0.3]})
    with pytest.raises(ValueError, match="--split"):
        convert(legacy, "PVC", "M", "Frozen")
    with pytest.raises(ValueError, match="sample-id-from-row-order"):
        convert(legacy, "PVC", "M", "Frozen", split="test")
    out = convert(legacy, "PVC", "M", "Frozen", split="test", sample_id_from_row_order=True)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.sample_id.tolist() == [0, 1, 0]
    assert out.y_score.tolist() == [0.2, 0.8, 0.3]


def test_convert_keeps_existing_split_and_sample_id():
    old_nb = pd.DataFrame({"patient_id": [5, 5], "sample_id": [7, 3], "split": ["VAL", "test"],
                           "label": [1, 0], "p_pvc": [0.6, 0.4]})
    out = convert(old_nb, "PVC", "M", "Frozen")
    assert out.split.tolist() == ["val", "test"] and out.sample_id.tolist() == [7, 3]
    with pytest.raises(ValueError, match="do not pass --split"):
        convert(old_nb, "PVC", "M", "Frozen", split="test")


def test_convert_legacy_synthetic_file_preserves_rows_and_auroc():
    raw = pd.read_csv(REPO / "legacy" / "pvc_predictions_legacy_synthetic.csv")
    out = convert(raw, "PVC", "HiMAE (legacy demo)", "Frozen", split="test", sample_id_from_row_order=True)
    assert len(out) == len(raw) == 11172
    assert roc_auc_score(out.y_true, out.y_score) == pytest.approx(roc_auc_score(raw.label, raw.p_pvc))


def test_released_canonical_example_is_valid():
    df = validate_predictions(pd.read_csv(REPO / "examples" / "synthetic_predictions.csv"))
    assert set(df.split) == {"val", "test"}
    for _, g in df[df.split == "test"].groupby(["task", "regime"]):
        wide_by_model(g)  # every model shares the same labelled windows
