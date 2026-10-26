"""Modular plotting utilities for efficiency and clinical-comparison figures.

This script is intended for the extended-figure analysis codebase.

All plotted values are loaded from user-provided CSV files.

Example
-------
python e-fig_efficiency.py \
    --efficiency-csv efficiency_results.csv \
    --clinical-csv clinical_results.csv \
    --output-dir figures \
    --baseline-model "HiMAE (1.2M)"

Expected efficiency CSV columns
-------------------------------
model,flops,memory,cpu_latency,cpu_throughput

Expected clinical CSV columns
-----------------------------
task,model,tp,sensitivity,balanced_accuracy,youden_index

Only columns that are present are plotted. Missing optional metrics are skipped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "flops",
    "memory",
    "cpu_latency",
    "cpu_throughput",
)

CLINICAL_METRICS = (
    "tp",
    "sensitivity",
    "balanced_accuracy",
    "youden_index",
)

METRIC_LABELS = {
    "flops": "FLOPs",
    "memory": "Memory",
    "cpu_latency": "CPU latency",
    "cpu_throughput": "CPU throughput",
    "tp": "True positives",
    "sensitivity": "Sensitivity",
    "balanced_accuracy": "Balanced accuracy",
    "youden_index": "Youden index",
}


# Visual defaults.
DEFAULT_FIGSIZE = (7.2, 5.6)
DEFAULT_DPI = 300
DEFAULT_BAR_ALPHA = 0.9
DEFAULT_GRID_ALPHA = 0.35
DEFAULT_ROTATION = 20


class InputFormatError(ValueError):
    """Raised when an input table does not contain the required structure."""


def configure_matplotlib(font_path: str | Path | None = None) -> None:
    """Configure plotting defaults with an optional custom font."""
    if font_path is not None:
        font_path = Path(font_path)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            rcParams["font.family"] = family

    rcParams["axes.unicode_minus"] = False
    rcParams["savefig.bbox"] = "tight"


def read_csv(path: str | Path, required_columns: Iterable[str]) -> pd.DataFrame:
    """Read a CSV and validate its required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    required = set(required_columns)
    missing = required.difference(frame.columns)
    if missing:
        raise InputFormatError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    return frame


def coerce_numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert selected columns to numeric and reject non-numeric values."""
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = converted.isna() & frame[column].notna()
        if invalid.any():
            bad_values = frame.loc[invalid, column].astype(str).unique().tolist()
            raise InputFormatError(
                f"Column '{column}' contains non-numeric values: {bad_values[:5]}"
            )
        frame[column] = converted
    return frame


def load_efficiency_results(path: str | Path) -> pd.DataFrame:
    """Load observed efficiency measurements."""
    frame = read_csv(path, required_columns=["model"])
    available = [metric for metric in DEFAULT_METRICS if metric in frame.columns]
    if not available:
        raise InputFormatError(
            "Efficiency CSV must contain at least one of: "
            + ", ".join(DEFAULT_METRICS)
        )

    frame = coerce_numeric_columns(frame, available)
    frame["model"] = frame["model"].astype(str).str.strip()

    if frame["model"].duplicated().any():
        duplicated = frame.loc[frame["model"].duplicated(False), "model"].tolist()
        raise InputFormatError(
            "Efficiency CSV must contain one observed row per model. "
            f"Duplicate models found: {sorted(set(duplicated))}"
        )

    return frame


def load_clinical_results(path: str | Path) -> pd.DataFrame:
    """Load observed clinical comparison metrics."""
    frame = read_csv(path, required_columns=["task", "model"])
    available = [metric for metric in CLINICAL_METRICS if metric in frame.columns]
    if not available:
        raise InputFormatError(
            "Clinical CSV must contain at least one of: "
            + ", ".join(CLINICAL_METRICS)
        )

    frame = coerce_numeric_columns(frame, available)
    frame["task"] = frame["task"].astype(str).str.strip()
    frame["model"] = frame["model"].astype(str).str.strip()

    keys = ["task", "model"]
    if frame.duplicated(keys).any():
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys]
        raise InputFormatError(
            "Clinical CSV must contain one row per task/model pair. "
            "Aggregate repeated runs before plotting.\n"
            + duplicates.to_string(index=False)
        )

    return frame


def compute_relative_values(
    frame: pd.DataFrame,
    metric: str,
    baseline_model: str,
) -> pd.Series:
    """Normalize an observed efficiency metric to a selected baseline model."""
    if metric not in frame.columns:
        raise KeyError(metric)

    baseline_rows = frame.loc[frame["model"] == baseline_model, metric]
    if baseline_rows.empty:
        raise InputFormatError(
            f"Baseline model '{baseline_model}' was not found in the efficiency CSV."
        )

    baseline_value = float(baseline_rows.iloc[0])
    if not np.isfinite(baseline_value) or baseline_value == 0:
        raise InputFormatError(
            f"Baseline value for '{metric}' must be finite and non-zero."
        )

    values = frame[metric].astype(float)
    return values / baseline_value


def sanitize_filename(value: str) -> str:
    """Convert a label into a filesystem-friendly stem."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "plot"


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply a clean publication-style axis treatment."""
    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=1,
        alpha=DEFAULT_GRID_ALPHA,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax: plt.Axes, bars, fmt: str) -> None:
    """Annotate bars with the values already being plotted."""
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_vertical_bar_plot(
    models: Sequence[str],
    values: Sequence[float],
    ylabel: str,
    output_path: str | Path,
    *,
    title: str | None = None,
    log_scale: bool = False,
    value_format: str = "{:.3g}",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Save one model-comparison bar plot."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(models):
        raise ValueError("models and values must be one-dimensional and equally sized")
    if not np.all(np.isfinite(values)):
        raise ValueError("Plot values must be finite")
    if log_scale and np.any(values <= 0):
        raise ValueError("Log-scale plots require all values to be positive")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))
    bars = ax.bar(x, values, alpha=DEFAULT_BAR_ALPHA, zorder=3)

    if log_scale:
        ax.set_yscale("log")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=DEFAULT_ROTATION, ha="right")
    if title:
        ax.set_title(title)

    style_axes(ax, grid_axis="y")
    annotate_bars(ax, bars, value_format)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_grouped_clinical_plot(
    frame: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    title: str | None = None,
    value_format: str = "{:.3g}",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot an observed clinical metric for all tasks and models."""
    if metric not in frame.columns:
        raise KeyError(metric)

    plot_frame = frame[["task", "model", metric]].dropna().copy()
    if plot_frame.empty:
        return

    pivot = plot_frame.pivot(index="task", columns="model", values=metric)
    tasks = pivot.index.tolist()
    models = pivot.columns.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(tasks), dtype=float)
    n_models = len(models)
    width = min(0.8 / max(n_models, 1), 0.35)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width

    for offset, model in zip(offsets, models):
        values = pivot[model].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            alpha=DEFAULT_BAR_ALPHA,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    value_format.format(value),
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    if title:
        ax.set_title(title)

    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_efficiency_suite(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    baseline_model: str | None = None,
    log_scale: bool = True,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Create absolute and, optionally, baseline-relative efficiency plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    models = frame["model"].tolist()

    for metric in DEFAULT_METRICS:
        if metric not in frame.columns:
            continue

        values = frame[metric].to_numpy(dtype=float)
        absolute_path = output_dir / f"{metric}_absolute.png"
        save_vertical_bar_plot(
            models,
            values,
            METRIC_LABELS[metric],
            absolute_path,
            title=f"{METRIC_LABELS[metric]} (absolute)",
            log_scale=log_scale,
            dpi=dpi,
        )
        created.append(absolute_path)

        if baseline_model is not None:
            relative = compute_relative_values(frame, metric, baseline_model)
            relative_path = output_dir / f"{metric}_relative.png"
            save_vertical_bar_plot(
                models,
                relative.to_numpy(dtype=float),
                f"{METRIC_LABELS[metric]} relative to {baseline_model}",
                relative_path,
                title=f"{METRIC_LABELS[metric]} (relative)",
                log_scale=log_scale,
                value_format="{:.2f}x",
                dpi=dpi,
            )
            created.append(relative_path)

    return created


def plot_clinical_suite(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Create one grouped plot per available observed clinical metric."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for metric in CLINICAL_METRICS:
        if metric not in frame.columns:
            continue
        output_path = output_dir / f"clinical_{sanitize_filename(metric)}.png"
        save_grouped_clinical_plot(
            frame,
            metric,
            output_path,
            title=METRIC_LABELS[metric],
            value_format="{:.0f}" if metric == "tp" else "{:.3f}",
            dpi=dpi,
        )
        created.append(output_path)

    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot model-efficiency and clinical-comparison results from CSV files."
        )
    )
    parser.add_argument(
        "--efficiency-csv",
        type=Path,
        help="CSV containing observed efficiency measurements.",
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        help="CSV containing observed clinical comparison metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help=(
            "Optional model used to derive relative efficiency values from the "
            "observed absolute measurements."
        ),
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional custom font file.",
    )
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Use a linear rather than logarithmic scale for efficiency plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output resolution in dots per inch.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.efficiency_csv is None and args.clinical_csv is None:
        raise SystemExit(
            "Provide at least one input: --efficiency-csv or --clinical-csv."
        )

    configure_matplotlib(args.font_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    if args.efficiency_csv is not None:
        efficiency = load_efficiency_results(args.efficiency_csv)
        created.extend(
            plot_efficiency_suite(
                efficiency,
                args.output_dir,
                baseline_model=args.baseline_model,
                log_scale=not args.linear_scale,
                dpi=args.dpi,
            )
        )

    if args.clinical_csv is not None:
        clinical = load_clinical_results(args.clinical_csv)
        created.extend(
            plot_clinical_suite(
                clinical,
                args.output_dir,
                dpi=args.dpi,
            )
        )

    for path in created:
        print(path.resolve())


if __name__ == "__main__":
    main()
