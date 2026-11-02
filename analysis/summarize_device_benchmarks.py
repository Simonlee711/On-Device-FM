#!/usr/bin/env python3
"""Summarize externally collected on-device latency/throughput logs (Table 1 on-device rows, Fig. 2).

This script does NOT measure anything. Latency and throughput were measured on a
Samsung Galaxy Watch 8 by a native Wear OS benchmarking application that is not
part of this repository, and the original device logs are not distributed. The
script only computes descriptive statistics and Welch tests from a CSV of those
already-measured values, so its outputs are only as valid as the supplied log.

Input (--runs): one row per repeated benchmark pass, exported from the device log:
    model,run,latency_s,throughput_wps
    latency_s      : forward-pass time per window (s)
    throughput_wps : windows processed per second over the pass
At least one of the two metric columns must be present. Any other column is
ignored and listed on stdout.

Optional input (--static): per-model constants, e.g. from analysis/model_complexity.py
--static-output (analytic) or values supplied for baselines whose code is not released:
    model,params_m,memory_mb

Per model: mean, median, 95th percentile, SD and a 95% t-interval of the mean
across passes. Mean differences against --reference-model use two-sided Welch
t-tests (unequal variances); no multiplicity adjustment.

Outputs:
    <output-dir>/device_summary.csv      model,metric,mean,median,p95,sd,ci_low,ci_high,n_runs,source
    <output-dir>/device_welch_tests.csv  model,metric,reference,mean_model,mean_reference,mean_diff,
                                         ratio_to_reference,t_stat,df,p_value
    <output-dir>/fig2_efficiency.csv     model,memory,cpu_latency,cpu_throughput,params
                                         -> plot/fig2_3.py --efficiency-csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_METRICS = {"latency_s": "cpu_latency", "throughput_wps": "cpu_throughput"}
STATIC_METRICS = {"memory_mb": "memory", "params_m": "params"}
SOURCE = "external device log (supplied input; not measured by this script)"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--static", type=Path, default=None)
    p.add_argument("--reference-model", default="HiMAE (1.2M)")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args(argv)


def describe(x: np.ndarray, confidence: float) -> dict[str, float]:
    x = x[np.isfinite(x)]
    n = x.size
    mean = float(np.mean(x)) if n else float("nan")
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    if n > 1:
        half = stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n)
        lo, hi = mean - half, mean + half
    else:
        lo = hi = float("nan")
    return {"mean": mean, "median": float(np.median(x)) if n else float("nan"),
            "p95": float(np.percentile(x, 95)) if n else float("nan"),
            "sd": sd, "ci_low": float(lo), "ci_high": float(hi), "n_runs": int(n)}


def summarize(runs: pd.DataFrame, reference: str, confidence: float = 0.95):
    runs = runs.copy()
    runs.columns = [str(c).strip().lower() for c in runs.columns]
    if "model" not in runs.columns:
        raise ValueError("Runs CSV must contain a 'model' column.")
    present = [c for c in RUN_METRICS if c in runs.columns]
    if not present:
        raise ValueError(f"Runs CSV contains none of {sorted(RUN_METRICS)}.")
    ignored = sorted(set(runs.columns) - set(RUN_METRICS) - {"model", "run"})
    if ignored:
        print(f"[info] ignoring columns not analyzed by this script: {ignored}")
    runs["model"] = runs["model"].astype(str).str.strip()
    models = list(dict.fromkeys(runs["model"]))

    summary, tests = [], []
    for col in present:
        values = {m: pd.to_numeric(runs.loc[runs.model == m, col], errors="coerce").dropna().to_numpy(dtype=float)
                  for m in models}
        for m in models:
            summary.append({"model": m, "metric": col, **describe(values[m], confidence), "source": SOURCE})
        if reference not in models:
            continue
        xr = values[reference]
        for m in models:
            xm = values[m]
            if m == reference or len(xr) < 2 or len(xm) < 2:
                continue
            res = stats.ttest_ind(xm, xr, equal_var=False)
            v1, v2 = xm.var(ddof=1) / len(xm), xr.var(ddof=1) / len(xr)
            dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(xm) - 1) + v2 ** 2 / (len(xr) - 1))
            tests.append({"model": m, "metric": col, "reference": reference,
                          "mean_model": xm.mean(), "mean_reference": xr.mean(),
                          "mean_diff": xm.mean() - xr.mean(), "ratio_to_reference": xm.mean() / xr.mean(),
                          "t_stat": float(res.statistic), "df": float(dof), "p_value": float(res.pvalue)})
    return pd.DataFrame(summary), pd.DataFrame(tests), models


def efficiency_table(summary: pd.DataFrame, models: list[str], static: pd.DataFrame | None) -> pd.DataFrame:
    eff = summary.pivot(index="model", columns="metric", values="mean").rename(columns=RUN_METRICS).reindex(models)
    if static is not None:
        st = static.copy()
        st.columns = [str(c).strip().lower() for c in st.columns]
        st["model"] = st["model"].astype(str).str.strip()
        st = st.set_index("model")[[c for c in STATIC_METRICS if c in st.columns]].rename(columns=STATIC_METRICS)
        eff = st.join(eff, how="outer").reindex(list(dict.fromkeys(models + list(st.index))))
    cols = [c for c in ("memory", "cpu_latency", "cpu_throughput", "params") if c in eff.columns]
    return eff[cols].rename_axis("model").reset_index()


def main(argv=None):
    args = parse_args(argv)
    summary, tests, models = summarize(pd.read_csv(args.runs), args.reference_model, args.confidence)
    static = pd.read_csv(args.static) if args.static is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "device_summary.csv", index=False)
    tests.to_csv(args.output_dir / "device_welch_tests.csv", index=False)
    efficiency_table(summary, models, static).to_csv(args.output_dir / "fig2_efficiency.csv", index=False)
    print(f"Summarized supplied device logs for {len(models)} models into {args.output_dir}")


if __name__ == "__main__":
    main()
