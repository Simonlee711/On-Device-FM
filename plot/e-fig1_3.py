"""Utilities for reproducing Extended Figures 1-3.

This script loads confusion-matrix operating-point CSV files, validates them,
selects a common model-score threshold, and renders the corresponding grid of
row-normalized confusion matrices.

Expected CSV columns
--------------------
threshold, task, model, regime, sensitivity, specificity, tp, fp, tn, fn

Example
-------
python e-fig1_3.py --data-dir ./data --threshold 0.05 --output-dir ./figures
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# Extended Figures 1-3.

REQUIRED_COLUMNS = {
    "threshold",
    "task",
    "model",
    "regime",
    "sensitivity",
    "specificity",
    "tp",
    "fp",
    "tn",
    "fn",
}

NUMERIC_COLUMNS = [
    "threshold",
    "sensitivity",
    "specificity",
    "tp",
    "fp",
    "tn",
    "fn",
]


@dataclass(frozen=True)
class FigureConfig:
    """Configuration for the Extended Figures 1-3 confusion-matrix plot."""

    threshold: float = 0.05
    threshold_tolerance: float = 0.0051
    strict_missing: bool = True

    row_order: Sequence[tuple[str, str]] = field(
        default_factory=lambda: (
            ("PVC", "Frozen"),
            ("PVC", "Finetuned"),
            ("Hypertension", "Frozen"),
            ("Hypertension", "Finetuned"),
        )
    )
    model_order: Sequence[str] = field(
        default_factory=lambda: (
            "HiMAE_1.2M",
            "PaPaGei_5.7M",
            "EfficientNet_7.8M",
            "MAE1D_6.7M",
            "MAE1D_110M",
        )
    )

    row_display_names: Mapping[tuple[str, str], str] = field(
        default_factory=lambda: {
            ("PVC", "Frozen"): "PVC\nFrozen",
            ("PVC", "Finetuned"): "PVC\nFine-tuned",
            ("Hypertension", "Frozen"): "Hypertension\nFrozen",
            ("Hypertension", "Finetuned"): "Hypertension\nFine-tuned",
        }
    )
    model_display_names: Mapping[str, str] = field(
        default_factory=lambda: {
            "HiMAE_1.2M": "HiMAE\n(1.2M)",
            "PaPaGei_5.7M": "PaPaGei\n(5.7M)",
            "EfficientNet_7.8M": "EfficientNet\n(7.8M)",
            "MAE1D_6.7M": "MAE1D\n(6.7M)",
            "MAE1D_110M": "MAE1D\n(110M)",
        }
    )


MODEL_ALIASES = {
    "HiMAE_1.2M": {"himae", "himae12m"},
    "PaPaGei_5.7M": {"papagei", "papagei57m"},
    "EfficientNet_7.8M": {"efficientnet", "efficientnet78m"},
    "MAE1D_6.7M": {"mae1d67m", "vitmae1d67m"},
    "MAE1D_110M": {"mae1d110m", "vitmae1d110m"},
}


CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "confusion",
    ["#750814", "#C66767", "#C66767", "#EFE5E0", "#F4D1D1"],
)

TEXT_COLOR = "#202124"
LIGHT_GRAY = "#F3F4F6"


def configure_plot_style() -> None:
    """Apply a compact, publication-friendly plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "figure.autolayout": True,
        }
    )


def normalize_token(value: object) -> str:
    """Lowercase and remove punctuation, spaces, underscores, and hyphens."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def canonicalize_model(value: object) -> str:
    key = normalize_token(value)
    for canonical_name, aliases in MODEL_ALIASES.items():
        if key in aliases:
            return canonical_name
    return str(value).strip()


def canonicalize_task(value: object) -> str:
    key = normalize_token(value)
    if key == "pvc":
        return "PVC"
    if key in {"hypertension", "htn", "hypertensionlab", "labhypertension"}:
        return "Hypertension"
    return str(value).strip()


def canonicalize_regime(value: object) -> str:
    key = normalize_token(value)
    if key == "frozen":
        return "Frozen"
    if key in {"finetuned", "finetune", "finetuning", "fullyfinetuned"}:
        return "Finetuned"
    if key == "supervised":
        return "Supervised"
    return str(value).strip()


def read_results(data_dir: Path) -> pd.DataFrame:
    """Load, normalize, validate, and deduplicate all CSV files in a folder."""
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {data_dir.resolve()}")

    tables: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        try:
            table = pd.read_csv(csv_path)
        except Exception as exc:
            warnings.warn(f"Could not read {csv_path.name}: {exc}")
            continue

        table.columns = [str(column).strip().lower() for column in table.columns]
        missing = REQUIRED_COLUMNS.difference(table.columns)
        if missing:
            warnings.warn(
                f"Skipping {csv_path.name}; missing columns: {sorted(missing)}"
            )
            continue

        table = table.copy()
        table["source_file"] = csv_path.name
        table["task"] = table["task"].map(canonicalize_task)
        table["model"] = table["model"].map(canonicalize_model)
        table["regime"] = table["regime"].map(canonicalize_regime)

        for column in NUMERIC_COLUMNS:
            table[column] = pd.to_numeric(table[column], errors="coerce")
        table = table.dropna(subset=NUMERIC_COLUMNS)

        if not table.empty:
            tables.append(table)

    if not tables:
        raise RuntimeError("No readable CSV contained the required columns.")

    results = pd.concat(tables, ignore_index=True, sort=False).rename(
        columns={"threshold": "score_threshold"}
    )

    dedup_columns = [
        "task",
        "model",
        "regime",
        "score_threshold",
        "sensitivity",
        "specificity",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    results = results.drop_duplicates(subset=dedup_columns, keep="first").reset_index(
        drop=True
    )

    curve_key = ["task", "model", "regime", "score_threshold"]
    conflicts = results.duplicated(subset=curve_key, keep=False)
    if conflicts.any():
        conflict_rows = results.loc[
            conflicts, curve_key + ["source_file", "tp", "fp", "tn", "fn"]
        ].sort_values(curve_key)
        raise RuntimeError(
            "Conflicting rows found for the same task/model/regime/threshold:\n"
            + conflict_rows.to_string(index=False)
        )

    return results


def add_derived_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Recompute all plotted rates from the integer confusion-matrix counts."""
    results = results.copy()

    for column in ("tp", "fp", "tn", "fn"):
        rounded = np.rint(results[column])
        if not np.allclose(results[column], rounded, atol=1e-9, rtol=0):
            raise ValueError(f"Column '{column}' contains non-integer counts.")
        results[column] = rounded.astype(np.int64)

    results["n_positive"] = results["tp"] + results["fn"]
    results["n_negative"] = results["tn"] + results["fp"]
    if (results["n_positive"] <= 0).any() or (results["n_negative"] <= 0).any():
        raise ValueError("Every curve row must contain positive and negative examples.")

    results["cm_sensitivity"] = results["tp"] / results["n_positive"]
    results["cm_specificity"] = results["tn"] / results["n_negative"]
    results["cm_fpr"] = results["fp"] / results["n_negative"]
    results["cm_balanced_accuracy"] = (
        results["cm_sensitivity"] + results["cm_specificity"]
    ) / 2.0
    results["cm_youden"] = results["cm_sensitivity"] + results["cm_specificity"] - 1.0

    results["reported_balanced_accuracy"] = (
        results["sensitivity"] + results["specificity"]
    ) / 2.0
    results["reported_youden"] = results["sensitivity"] + results["specificity"] - 1.0

    totals = results.groupby(["task", "model", "regime"]).agg(
        n_positive_min=("n_positive", "min"),
        n_positive_max=("n_positive", "max"),
        n_negative_min=("n_negative", "min"),
        n_negative_max=("n_negative", "max"),
    )
    inconsistent = totals.loc[
        (totals["n_positive_min"] != totals["n_positive_max"])
        | (totals["n_negative_min"] != totals["n_negative_max"])
    ]
    if not inconsistent.empty:
        raise ValueError(
            "Class totals change across threshold rows for these curves:\n"
            + inconsistent.to_string()
        )

    return results


def validate_expected_panels(results: pd.DataFrame, config: FigureConfig) -> None:
    """Optionally require every configured task/regime/model panel to exist."""
    available = {
        (task, regime, model)
        for task, regime, model in results[["task", "regime", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    missing = [
        (task, regime, model)
        for task, regime in config.row_order
        for model in config.model_order
        if (task, regime, model) not in available
    ]

    if missing and config.strict_missing:
        formatted = "\n".join(
            f"  - task={task}, regime={regime}, model={model}"
            for task, regime, model in missing
        )
        raise RuntimeError(
            "Missing expected task/model/regime combinations:\n"
            f"{formatted}\n\n"
            "Add the missing files, change the figure configuration, or use "
            "--allow-missing."
        )


def get_curve(
    results: pd.DataFrame, task: str, regime: str, model: str
) -> pd.DataFrame | None:
    curve = results.loc[
        (results["task"] == task)
        & (results["regime"] == regime)
        & (results["model"] == model)
    ].copy()
    if curve.empty:
        return None
    return curve.sort_values("score_threshold").reset_index(drop=True)


def select_nearest_threshold(
    curve: pd.DataFrame, requested_threshold: float, tolerance: float
) -> pd.Series:
    """Select the row nearest a requested model-score threshold."""
    curve = curve.copy()
    curve["_threshold_distance"] = np.abs(
        curve["score_threshold"] - requested_threshold
    )
    selected = curve.sort_values(
        ["_threshold_distance", "score_threshold"], ascending=[True, True]
    ).iloc[0]

    if selected["_threshold_distance"] > tolerance:
        raise ValueError(
            f"No threshold sufficiently close to {requested_threshold:.4f}; "
            f"nearest available threshold is {selected['score_threshold']:.4f}."
        )
    return selected


def select_operating_point(
    results: pd.DataFrame,
    task: str,
    regime: str,
    model: str,
    config: FigureConfig,
) -> pd.Series | None:
    curve = get_curve(results, task, regime, model)
    if curve is None:
        return None
    return select_nearest_threshold(
        curve, requested_threshold=config.threshold, tolerance=config.threshold_tolerance
    )


def create_confusion_matrix(row: pd.Series) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]] with actual classes on rows."""
    return np.array(
        [[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]],
        dtype=np.int64,
    )


def normalize_by_actual_class(confusion_matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix to percentages."""
    row_totals = confusion_matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        confusion_matrix,
        row_totals,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_totals != 0,
    ) * 100.0

    if not np.allclose(percentages.sum(axis=1), 100.0, atol=1e-9):
        raise AssertionError("Each actual-class row must sum to 100%.")
    return percentages


def make_annotations(counts: np.ndarray, percentages: np.ndarray) -> np.ndarray:
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]], dtype=object)
    annotations = np.empty(counts.shape, dtype=object)
    for row_index in range(2):
        for column_index in range(2):
            annotations[row_index, column_index] = (
                f"{cell_names[row_index, column_index]}\n"
                f"{counts[row_index, column_index]:,}\n"
                f"{percentages[row_index, column_index]:.1f}%"
            )
    return annotations


def build_summary_row(
    selected: pd.Series, task: str, regime: str, model: str, config: FigureConfig
) -> dict[str, object]:
    return {
        "task": task,
        "regime": regime,
        "model": model,
        "requested_threshold": config.threshold,
        "selected_threshold": float(selected["score_threshold"]),
        "fpr_from_counts": float(selected["cm_fpr"]),
        "sensitivity_from_counts": float(selected["cm_sensitivity"]),
        "specificity_from_counts": float(selected["cm_specificity"]),
        "balanced_accuracy_from_counts": float(selected["cm_balanced_accuracy"]),
        "youden_from_counts": float(selected["cm_youden"]),
        "reported_sensitivity": float(selected["sensitivity"]),
        "reported_specificity": float(selected["specificity"]),
        "reported_balanced_accuracy": float(selected["reported_balanced_accuracy"]),
        "reported_youden": float(selected["reported_youden"]),
        "tn": int(selected["tn"]),
        "fp": int(selected["fp"]),
        "fn": int(selected["fn"]),
        "tp": int(selected["tp"]),
        "n_negative": int(selected["n_negative"]),
        "n_positive": int(selected["n_positive"]),
        "source_file": selected["source_file"],
    }


def plot_confusion_grid(
    results: pd.DataFrame,
    config: FigureConfig,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Create the full confusion-matrix grid and return its summary table."""
    configure_plot_style()
    validate_expected_panels(results, config)

    n_rows = len(config.row_order)
    n_columns = len(config.model_order)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(17.2, 13.4),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    summary_rows: list[dict[str, object]] = []

    for row_index, (task, regime) in enumerate(config.row_order):
        for column_index, model in enumerate(config.model_order):
            ax = axes[row_index, column_index]
            selected = select_operating_point(results, task, regime, model, config)

            if selected is None:
                ax.set_facecolor(LIGHT_GRAY)
                ax.text(
                    0.5,
                    0.5,
                    "Missing CSV",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#777777",
                    fontweight="semibold",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue

            counts = create_confusion_matrix(selected)
            percentages = normalize_by_actual_class(counts)
            annotations = make_annotations(counts, percentages)

            if not np.isclose(percentages[0, 0] / 100.0, selected["cm_specificity"]):
                raise AssertionError("Plotted TN percentage does not equal specificity.")
            if not np.isclose(percentages[0, 1] / 100.0, selected["cm_fpr"]):
                raise AssertionError("Plotted FP percentage does not equal FPR.")

            sns.heatmap(
                percentages,
                ax=ax,
                cmap=CONFUSION_CMAP,
                vmin=0,
                vmax=100,
                square=True,
                cbar=False,
                linewidths=1.7,
                linecolor="white",
                annot=annotations,
                fmt="",
                annot_kws={"fontsize": 9.8, "fontweight": "semibold"},
            )

            for text_object, percentage in zip(ax.texts, percentages.ravel()):
                text_object.set_color("black" if percentage >= 55.0 else TEXT_COLOR)

            ax.set_xlabel("")
            ax.set_ylabel("")

            if row_index == 0:
                ax.set_title(
                    config.model_display_names.get(model, model),
                    fontsize=13,
                    fontweight="bold",
                    color=TEXT_COLOR,
                    pad=13,
                )

            ax.set_xticklabels(["Pred. negative", "Pred. positive"], rotation=0)
            ax.set_yticklabels(["Actual negative", "Actual positive"], rotation=0)

            if row_index != n_rows - 1:
                ax.set_xticklabels([])
            if column_index != 0:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", length=0, pad=5)
            ax.text(
                0.5,
                -0.115,
                (
                    f"Score threshold {selected['score_threshold']:.2f}  |  "
                    f"Sens. {selected['cm_sensitivity']:.3f}\n"
                    f"Spec. {selected['cm_specificity']:.3f}  |  "
                    f"BA {selected['cm_balanced_accuracy']:.3f}"
                ),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.9,
                linespacing=1.25,
                color=TEXT_COLOR,
            )

            if column_index == 0:
                ax.annotate(
                    config.row_display_names.get((task, regime), f"{task}\n{regime}"),
                    xy=(-0.67, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=12.5,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )

            summary_rows.append(build_summary_row(selected, task, regime, model, config))

    fig.suptitle(
        f"Window-level confusion matrices at score threshold = {config.threshold:.2f}",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.995,
    )
    fig.text(
        0.5,
        -0.020,
        "Predicted class",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="semibold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.012,
        0.5,
        "Actual class",
        ha="left",
        va="center",
        rotation=90,
        fontsize=13,
        fontweight="semibold",
        color=TEXT_COLOR,
    )
    plt.subplots_adjust(
        left=0.115,
        right=0.992,
        top=0.925,
        bottom=0.090,
        wspace=0.22,
        hspace=0.47,
    )

    return fig, pd.DataFrame(summary_rows)


def save_outputs(
    fig: plt.Figure,
    summary: pd.DataFrame,
    output_dir: Path,
    stem: str = "extended_figure1_3",
) -> tuple[Path, Path, Path]:
    """Save PNG, PDF, and CSV summary outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    csv_path = output_dir / f"{stem}_summary.csv"

    summary.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return png_path, pdf_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices for Extended Figures 1-3."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./figures"),
        help="Directory for generated figures and summary CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Common model-score threshold used for all panels.",
    )
    parser.add_argument(
        "--threshold-tolerance",
        type=float,
        default=0.0051,
        help="Maximum distance from the requested threshold.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Render missing panels instead of raising an error.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save outputs without opening the matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for generating Extended Figures 1-3."""
    args = parse_args()
    config = FigureConfig(
        threshold=args.threshold,
        threshold_tolerance=args.threshold_tolerance,
        strict_missing=not args.allow_missing,
    )

    results = add_derived_metrics(read_results(args.data_dir))
    fig, summary = plot_confusion_grid(results, config)
    png_path, pdf_path, csv_path = save_outputs(fig, summary, args.output_dir)

    if not summary.empty:
        display_columns = [
            "task",
            "regime",
            "model",
            "selected_threshold",
            "fpr_from_counts",
            "sensitivity_from_counts",
            "specificity_from_counts",
            "balanced_accuracy_from_counts",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        print("\nSelected operating points")
        print("-" * 120)
        print(summary[display_columns].to_string(index=False))

    print(f"\nSaved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")
    print(f"Saved summary CSV: {csv_path.resolve()}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
