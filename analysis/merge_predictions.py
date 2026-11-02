#!/usr/bin/env python3
"""Concatenate per-model canonical prediction CSVs into one analysis table.

Every input must already follow the canonical schema (analysis/prediction_schema.py);
convert legacy exports first with analysis/convert_predictions.py. The merged table
is re-validated, so a window exported twice for the same task/model/regime/split
is rejected rather than silently duplicated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prediction_schema import CANONICAL_COLUMNS, load_predictions, validate_predictions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/all_predictions.csv"))
    a = p.parse_args(argv)
    frames = [load_predictions(path)[CANONICAL_COLUMNS] for path in a.inputs]
    out = validate_predictions(pd.concat(frames, ignore_index=True), "Merged predictions")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out):,} rows to {a.output}")


if __name__ == "__main__":
    main()
