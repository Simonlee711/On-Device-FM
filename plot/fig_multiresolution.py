#!/usr/bin/env python3
"""Plot test AUROC by HiMAE encoder stage (temporal resolution).

Input: one or more multires_probe_summary.csv files written by
pvc/downstream_eval/multiresolution_probe.py (rows for several tasks may be
concatenated). One panel per task: per-stage test AUROC with participant-level
bootstrap 95% CIs, x-axis labelled by stage and receptive field; the
all-stages concatenation is drawn as a dashed reference line and the
validation-selected stage is highlighted.

Outputs:
    <output-dir>/multiresolution_probe.png
    <output-dir>/multiresolution_probe.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = {"task", "level", "stage", "receptive_field_ms", "test_auroc", "ci_low", "ci_high"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("figures"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.concat([pd.read_csv(p) for p in args.input], ignore_index=True)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    tasks = list(dict.fromkeys(df["task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.8 * len(tasks), 4.0), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        d = df[df.task == task]
        st = d[d.level != "all_stages"].copy()
        st["stage_num"] = st["stage"].astype(int)
        st = st.sort_values("stage_num")
        x = np.arange(len(st))
        y = st["test_auroc"].to_numpy(float)
        err = np.vstack([y - st["ci_low"].to_numpy(float), st["ci_high"].to_numpy(float) - y])
        ax.errorbar(x, y, yerr=np.clip(err, 0, None), fmt="o-", color="#7a0019", capsize=3, lw=2, label="Single stage")
        if "selected_on_validation" in st.columns:
            sel = st["selected_on_validation"].astype(str).str.lower().eq("true").to_numpy()
            ax.scatter(x[sel], y[sel], s=160, facecolors="none", edgecolors="black", lw=1.5,
                       label="Selected on validation", zorder=3)
        allr = d[d.level == "all_stages"]
        if not allr.empty:
            ax.axhline(allr["test_auroc"].item(), ls="--", color="grey", label="All stages (concat)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}\n{rf:.0f} ms" for s, rf in zip(st["stage_num"], st["receptive_field_ms"])])
        ax.set_xlabel("Encoder stage / receptive field")
        ax.set_ylabel("Test AUROC")
        ax.set_title(task)
        ax.grid(True, ls="--", alpha=0.4)
    axes[0][-1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = args.output_dir / f"multiresolution_probe.{ext}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
