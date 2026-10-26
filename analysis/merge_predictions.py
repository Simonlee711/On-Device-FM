#!/usr/bin/env python3
"""Concatenate standardized prediction CSVs into one analysis table."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
REQ={'task','model','regime','split','y_true','y_score'}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs', nargs='+', type=Path, required=True)
p.add_argument('--output', type=Path, default=Path('results/all_predictions.csv'))
a=p.parse_args()
frames=[]
for path in a.inputs:
    d=pd.read_csv(path); missing=REQ.difference(d.columns)
    if missing: raise ValueError(f'{path} missing {sorted(missing)}')
    frames.append(d)
out=pd.concat(frames,ignore_index=True,sort=False)
a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
print(f'Wrote {len(out):,} rows to {a.output}')
