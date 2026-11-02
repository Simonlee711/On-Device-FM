#!/usr/bin/env python3
"""Convert a legacy per-model prediction export to the canonical schema.

Legacy exports (the earlier demo notebook, legacy/pvc_predictions_legacy_synthetic.csv)
use per-model columns such as

    patient_id,[sample_id,][split,]label,p_pvc

and carry no task / model / regime. This utility adds those fields and renames
columns to the canonical schema read by every statistical script:

    task,model,regime,split,person_id,sample_id,y_true,y_score

Missing fields are never invented silently:

* no split column  -> --split must be given (applied to every row);
* no sample_id     -> --sample-id-from-row-order must be given. sample_id is then
                      the position of each row within its participant, in file
                      order. This is only a valid pairing key if every model's
                      export lists each participant's windows in the same order;
                      otherwise paired analyses will compare different windows.

Example (the legacy synthetic file; it is a single random-window test split):

    python analysis/convert_predictions.py \
        --input legacy/pvc_predictions_legacy_synthetic.csv \
        --task PVC --model "HiMAE (legacy demo)" --regime Frozen \
        --split test --sample-id-from-row-order \
        --output results/legacy_predictions_canonical.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prediction_schema import (CANONICAL_COLUMNS, normalize_columns, validate_predictions,
                               window_ids_within_participant)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--regime", required=True)
    p.add_argument("--split", default=None, help="Split for every row; only allowed when the input has no split column.")
    p.add_argument("--sample-id-from-row-order", action="store_true",
                   help="Assign sample_id by row order within participant (input has no sample_id).")
    p.add_argument("--person-column", default="patient_id")
    p.add_argument("--sample-column", default="sample_id")
    p.add_argument("--split-column", default="split")
    p.add_argument("--label-column", default="label")
    p.add_argument("--score-column", default="p_pvc")
    return p.parse_args(argv)


def convert(df: pd.DataFrame, task: str, model: str, regime: str, split: str | None = None,
            sample_id_from_row_order: bool = False, person_column: str = "patient_id",
            sample_column: str = "sample_id", split_column: str = "split",
            label_column: str = "label", score_column: str = "p_pvc") -> pd.DataFrame:
    df = normalize_columns(df)
    person_column, sample_column, split_column, label_column, score_column = (
        c.strip().lower() for c in (person_column, sample_column, split_column, label_column, score_column))
    missing = [c for c in (person_column, label_column, score_column) if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}")

    if split_column in df.columns:
        if split is not None:
            raise ValueError(f"Input already has a '{split_column}' column; do not pass --split.")
        split_values = df[split_column]
    elif split is None:
        raise ValueError(f"Input has no '{split_column}' column; pass --split (e.g. test).")
    else:
        split_values = split

    if sample_column in df.columns:
        if sample_id_from_row_order:
            raise ValueError(f"Input already has a '{sample_column}' column; do not pass --sample-id-from-row-order.")
        sample_ids = df[sample_column]
    elif sample_id_from_row_order:
        sample_ids = window_ids_within_participant(df[person_column].to_numpy())
    else:
        raise ValueError(f"Input has no '{sample_column}' column, so windows cannot be paired across models. "
                         "Pass --sample-id-from-row-order only if every model export lists each "
                         "participant's windows in the same order.")

    out = pd.DataFrame({
        "task": task, "model": model, "regime": regime, "split": split_values,
        "person_id": df[person_column], "sample_id": sample_ids,
        "y_true": df[label_column], "y_score": df[score_column],
    }, columns=CANONICAL_COLUMNS)
    return validate_predictions(out, "Converted predictions")


def main(argv=None):
    a = parse_args(argv)
    out = convert(pd.read_csv(a.input), a.task, a.model, a.regime, a.split, a.sample_id_from_row_order,
                  a.person_column, a.sample_column, a.split_column, a.label_column, a.score_column)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out):,} canonical rows to {a.output}")


if __name__ == "__main__":
    main()
