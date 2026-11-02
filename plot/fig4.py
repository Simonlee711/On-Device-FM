"""
Modular plotting code for the few-shot extended figure.

All plotted values are read from a long-form input CSV.

Expected long-form CSV columns
------------------------------
Required:
    dataset   : dataset/task name (e.g., PVC, HTN)
    model     : model display name
    shots     : few-shot setting (numeric or string such as "all")
    auc       : AUROC/AUC value

Optional:
    run       : replicate/run identifier. If multiple runs are available for a
                dataset/model/shots combination, the script plots the mean with
                the interquartile range (25th-75th percentile) across runs.

Example rows:
    dataset,model,shots,run,auc
    PVC,HiMAE (1.2M),0,0,0.80
    PVC,HiMAE (1.2M),0,1,0.81
    PVC,HiMAE (1.2M),256,0,0.88

Usage
-----
python e-fig_fewshot.py --input results.csv --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"dataset", "model", "shots", "auc"}


# Visual defaults.
DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot few-shot AUROC curves from a long-form results CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the few-shot results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory in which figures are written.",
    )
    parser.add_argument(
        "--metric-label",
        default="AUROC",
        help="Y-axis label.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Suppress interquartile-range bars when replicate runs are present.",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate few-shot results."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    df = pd.read_csv(path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["shots"] = df["shots"].astype(str).str.strip()
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")

    if "run" in df.columns:
        df["run"] = df["run"].astype(str).str.strip()

    if df["auc"].isna().any():
        bad_rows = df.index[df["auc"].isna()].tolist()
        raise ValueError(f"Non-numeric AUC values found at rows: {bad_rows}")

    if not df["auc"].between(0.0, 1.0).all():
        raise ValueError("All AUC values must lie in [0, 1].")

    return df


def _shot_sort_key(value: str) -> tuple[int, float | str]:
    """Sort numeric shot counts before non-numeric labels."""
    text = str(value).strip()
    try:
        return (0, float(text))
    except ValueError:
        return (1, text.lower())


def infer_shot_order(values: Iterable[str]) -> list[str]:
    """Infer a stable x-axis order from shot labels."""
    unique = list(dict.fromkeys(str(value) for value in values))
    return sorted(unique, key=_shot_sort_key)


def summarize_results(
    df: pd.DataFrame,
    include_error_bars: bool = True,
) -> pd.DataFrame:
    """
    Summarize values for each dataset, model, and shot setting.

    If a ``run`` column is present, repeated runs are summarized by their mean.
    The interquartile range (25th-75th percentile) is computed when at least two
    runs are available.
    """
    group_columns = ["dataset", "model", "shots"]

    if "run" not in df.columns:
        duplicates = df.duplicated(subset=group_columns, keep=False)
        if duplicates.any():
            duplicate_rows = df.loc[duplicates, group_columns].drop_duplicates()
            raise ValueError(
                "Multiple values exist for the same dataset/model/shots setting "
                "but the CSV has no 'run' column to identify replicates:\n"
                + duplicate_rows.to_string(index=False)
            )

        summary = df[group_columns + ["auc"]].copy()
        summary = summary.rename(columns={"auc": "mean_auc"})
        summary["n_runs"] = 1
        summary["q25_auc"] = np.nan
        summary["q75_auc"] = np.nan
        return summary

    grouped = df.groupby(group_columns, sort=False)["auc"]
    summary = grouped.agg(
        mean_auc="mean",
        n_runs="count",
        q25_auc=lambda v: v.quantile(0.25),
        q75_auc=lambda v: v.quantile(0.75),
    ).reset_index()

    no_interval = (summary["n_runs"] < 2) | (not include_error_bars)
    summary.loc[no_interval, ["q25_auc", "q75_auc"]] = np.nan
    return summary


def compute_ylim(values: np.ndarray, padding_fraction: float = 0.05) -> tuple[float, float]:
    """Choose y-axis limits from plotted values."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return 0.0, 1.0

    minimum = float(finite.min())
    maximum = float(finite.max())
    span = maximum - minimum

    if span == 0:
        span = max(abs(maximum), 1.0) * 0.05

    padding = span * padding_fraction
    return max(0.0, minimum - padding), min(1.0, maximum + padding)


def plot_dataset(
    summary: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    metric_label: str = "AUROC",
    dpi: int = 300,
    show: bool = False,
) -> tuple[Path, Path]:
    """Plot the few-shot curve for one dataset."""
    subset = summary.loc[summary["dataset"] == dataset].copy()
    if subset.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")

    shot_order = infer_shot_order(subset["shots"])
    shot_to_x = {shot: index for index, shot in enumerate(shot_order)}

    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    models = list(dict.fromkeys(subset["model"]))

    for model_index, model in enumerate(models):
        model_df = subset.loc[subset["model"] == model].copy()
        model_df["x"] = model_df["shots"].map(shot_to_x)
        model_df = model_df.sort_values("x")

        marker = DEFAULT_MARKERS[model_index % len(DEFAULT_MARKERS)]
        x = model_df["x"].to_numpy(dtype=float)
        y = model_df["mean_auc"].to_numpy(dtype=float)

        ax.plot(x, y, marker=marker, linewidth=2.0, markersize=6, label=model)

        q25 = model_df["q25_auc"].to_numpy(dtype=float)
        q75 = model_df["q75_auc"].to_numpy(dtype=float)
        if np.isfinite(q25).any():
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([np.clip(y - q25, 0, None), np.clip(q75 - y, 0, None)]),
                fmt="none",
                elinewidth=1.0,
                capsize=3,
            )

    ax.set_xticks(range(len(shot_order)))
    ax.set_xticklabels(shot_order)
    ax.set_xlabel("Shots")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{dataset}: Few-Shot Learning Performance")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ymin, ymax = compute_ylim(subset["mean_auc"].to_numpy())
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_dataset_name = "_".join(dataset.lower().split())
    png_path = output_dir / f"fewshot_{safe_dataset_name}.png"
    pdf_path = output_dir / f"fewshot_{safe_dataset_name}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return png_path, pdf_path


def plot_all_datasets(
    df: pd.DataFrame,
    output_dir: Path,
    metric_label: str = "AUROC",
    dpi: int = 300,
    show: bool = False,
    include_error_bars: bool = True,
) -> list[tuple[Path, Path]]:
    """Create one few-shot figure per dataset."""
    summary = summarize_results(
        df,
        include_error_bars=include_error_bars,
    )

    outputs = []
    for dataset in dict.fromkeys(summary["dataset"]):
        outputs.append(
            plot_dataset(
                summary=summary,
                dataset=dataset,
                output_dir=output_dir,
                metric_label=metric_label,
                dpi=dpi,
                show=show,
            )
        )

    return outputs


def main() -> None:
    """Run the plotting pipeline."""
    args = parse_args()
    results = load_results(args.input)
    outputs = plot_all_datasets(
        df=results,
        output_dir=args.output_dir,
        metric_label=args.metric_label,
        dpi=args.dpi,
        show=args.show,
        include_error_bars=not args.no_error_bars,
    )

    for png_path, pdf_path in outputs:
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
