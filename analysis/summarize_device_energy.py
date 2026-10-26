#!/usr/bin/env python3
"""Summarize repeated on-device benchmark runs: latency, throughput, energy and discharge.

Input (--runs): one row per timed / telemetry-synchronized benchmark pass:
    model,run,latency_s,throughput_wps,energy_j
Every metric column is optional; any subset may be present. Optional discharge
columns (per run): battery_start_pct,battery_end_pct,duration_s,n_windows.

Optional input (--static): analytically computed, per-model constants:
    model,params_m,memory_mb,flops_gflops

Per model the script reports the mean, median, 95th percentile, SD and a 95%
t-based confidence interval of the mean across repeated runs (Table 1 on-device
rows). Between-model mean differences against --reference-model use two-sided
Welch t-tests (unequal variances); no multiplicity adjustment is applied.
When discharge columns are present, battery drain is summarized as %/hour and
windows per 1% battery.

Outputs:
    <output-dir>/device_summary.csv        long form: model,metric,mean,median,p95,sd,ci_low,ci_high,n_runs
    <output-dir>/device_welch_tests.csv    model,metric,reference,mean_diff,ratio,t_stat,df,p_value
    <output-dir>/device_discharge.csv      (only if discharge columns are present)
    <output-dir>/fig2_efficiency.csv       model,flops,memory,cpu_latency,cpu_throughput,energy
                                           -> consumed by plot/fig2_3.py --efficiency-csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_METRICS = {"latency_s": "cpu_latency", "throughput_wps": "cpu_throughput", "energy_j": "energy"}
STATIC_METRICS = {"flops_gflops": "flops", "memory_mb": "memory", "params_m": "params"}
DISCHARGE = ["battery_start_pct", "battery_end_pct", "duration_s", "n_windows"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--static", type=Path, default=None)
    p.add_argument("--reference-model", default="HiMAE (1.2M)")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


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


def main():
    args = parse_args()
    runs = pd.read_csv(args.runs)
    runs.columns = [str(c).strip().lower() for c in runs.columns]
    if "model" not in runs.columns:
        raise ValueError("Runs CSV must contain a 'model' column.")
    runs["model"] = runs["model"].astype(str).str.strip()
    present = [c for c in RUN_METRICS if c in runs.columns]
    if not present:
        raise ValueError(f"Runs CSV contains none of {sorted(RUN_METRICS)}.")
    models = list(dict.fromkeys(runs["model"]))

    summary, tests = [], []
    for col in present:
        for m in models:
            x = pd.to_numeric(runs.loc[runs.model == m, col], errors="coerce").to_numpy(dtype=float)
            summary.append({"model": m, "metric": col, **describe(x, args.confidence)})
        ref = args.reference_model
        if ref not in models:
            continue
        xr = pd.to_numeric(runs.loc[runs.model == ref, col], errors="coerce").dropna().to_numpy(dtype=float)
        for m in models:
            if m == ref:
                continue
            xm = pd.to_numeric(runs.loc[runs.model == m, col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(xr) < 2 or len(xm) < 2:
                continue
            res = stats.ttest_ind(xm, xr, equal_var=False)
            v1, v2 = xm.var(ddof=1) / len(xm), xr.var(ddof=1) / len(xr)
            dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(xm) - 1) + v2 ** 2 / (len(xr) - 1))
            tests.append({"model": m, "metric": col, "reference": ref,
                          "mean_model": xm.mean(), "mean_reference": xr.mean(),
                          "mean_diff": xm.mean() - xr.mean(), "ratio_to_reference": xm.mean() / xr.mean(),
                          "t_stat": float(res.statistic), "df": float(dof), "p_value": float(res.pvalue)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame(summary)
    summ.to_csv(args.output_dir / "device_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(args.output_dir / "device_welch_tests.csv", index=False)

    if set(DISCHARGE).issubset(runs.columns):
        d = runs.copy()
        d["drain_pct"] = d["battery_start_pct"] - d["battery_end_pct"]
        d["drain_pct_per_hour"] = d["drain_pct"] / (d["duration_s"] / 3600.0)
        d["windows_per_pct"] = np.where(d["drain_pct"] > 0, d["n_windows"] / d["drain_pct"].where(d["drain_pct"] > 0, 1), np.nan)
        rows = []
        for m in models:
            g = d[d.model == m]
            for col in ("drain_pct_per_hour", "windows_per_pct"):
                rows.append({"model": m, "metric": col, **describe(g[col].to_numpy(dtype=float), args.confidence)})
        pd.DataFrame(rows).to_csv(args.output_dir / "device_discharge.csv", index=False)

    # Figure 2 / Table 1 efficiency table: means of repeated runs + static constants.
    eff = summ.pivot(index="model", columns="metric", values="mean").rename(columns=RUN_METRICS)
    eff = eff.reindex(models)
    if args.static is not None:
        st = pd.read_csv(args.static)
        st.columns = [str(c).strip().lower() for c in st.columns]
        st["model"] = st["model"].astype(str).str.strip()
        st = st.set_index("model")[[c for c in STATIC_METRICS if c in st.columns]].rename(columns=STATIC_METRICS)
        eff = st.join(eff, how="outer").reindex(list(dict.fromkeys(models + list(st.index))))
    cols = [c for c in ("flops", "memory", "cpu_latency", "cpu_throughput", "energy", "params") if c in eff.columns]
    eff[cols].reset_index().rename(columns={"index": "model"}).to_csv(args.output_dir / "fig2_efficiency.csv", index=False)
    print(f"Wrote device summaries for {len(models)} models to {args.output_dir}")


if __name__ == "__main__":
    main()
