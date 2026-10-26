#!/usr/bin/env python3
"""Prepare the efficiency CSV consumed by plot/fig2_3.py.

The script accepts already measured device/model metrics. Repeated benchmark rows
are reduced to one row per model by the requested aggregation. It does not run
on-device benchmarking itself.

Canonical output columns:
    model,flops,memory,cpu_latency,cpu_throughput,energy
Only columns present in the source are emitted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ALIASES = {
    "model": ["model", "model_name"],
    "flops": ["flops", "gflops", "flops_gflops"],
    "memory": ["memory", "memory_mb", "model_memory_mb", "weight_memory_mb"],
    "cpu_latency": ["cpu_latency", "cpu_latency_s", "latency", "latency_s"],
    "cpu_throughput": ["cpu_throughput", "cpu_throughput_wps", "throughput", "windows_per_second"],
    "energy": ["energy", "energy_j", "energy_per_window_j"],
}


def find_column(columns, candidates):
    lut = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lut:
            return lut[candidate.lower()]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Raw or summary benchmark CSV.")
    p.add_argument("--output", type=Path, default=Path("results/fig2_efficiency.csv"))
    p.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    mapping = {name: find_column(df.columns, candidates) for name, candidates in ALIASES.items()}
    if mapping["model"] is None:
        raise ValueError("Could not identify a model column.")

    numeric = [k for k in ("flops", "memory", "cpu_latency", "cpu_throughput", "energy") if mapping[k] is not None]
    if not numeric:
        raise ValueError("No recognized efficiency metric columns were found.")

    work = pd.DataFrame({"model": df[mapping["model"]].astype(str).str.strip()})
    for key in numeric:
        work[key] = pd.to_numeric(df[mapping[key]], errors="raise")

    agg = "mean" if args.aggregate == "mean" else "median"
    out = work.groupby("model", sort=False, as_index=False).agg({k: agg for k in numeric})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} model rows to {args.output}")


if __name__ == "__main__":
    main()
