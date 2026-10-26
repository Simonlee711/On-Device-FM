"""
Modular ROC plotting code for the extended figure.

Expected long-form CSV columns
------------------------------
Required:
    task      : task or dataset name
    model     : model display name
    y_true    : binary ground-truth label (0 or 1)
    y_score   : model score or probability for the positive class

Optional:
    sample_id : sample/window identifier
    person_id : participant identifier

Usage
-----
python e-fig_roc.py --input predictions.csv --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


REQUIRED_COLUMNS = {"task", "model", "y_true", "y_score"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot ROC curves from model prediction scores."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a long-form CSV containing labels and prediction scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory in which figures are written.",
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
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate prediction data."""
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
    df["task"] = df["task"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")

    if df[["y_true", "y_score"]].isna().any().any():
        bad_rows = df.index[df[["y_true", "y_score"]].isna().any(axis=1)].tolist()
        raise ValueError(f"Non-numeric labels or scores found at rows: {bad_rows}")

    labels = set(df["y_true"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError("Column 'y_true' must contain only binary labels 0 and 1.")

    return df


def compute_roc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ROC coordinates and AUROC for each task/model pair."""
    rows = []

    for (task, model), group in df.groupby(["task", "model"], sort=False):
        if group["y_true"].nunique() < 2:
            raise ValueError(
                f"Task/model pair ({task}, {model}) does not contain both classes."
            )

        fpr, tpr, _ = roc_curve(group["y_true"], group["y_score"])
        auroc = auc(fpr, tpr)

        rows.append(
            {
                "task": task,
                "model": model,
                "fpr": fpr,
                "tpr": tpr,
                "auroc": float(auroc),
                "n_samples": int(len(group)),
            }
        )

    return pd.DataFrame(rows)


def make_safe_name(value: str) -> str:
    """Convert a label into a filesystem-friendly name."""
    return "_".join(str(value).strip().lower().split())


def plot_task(
    roc_table: pd.DataFrame,
    task: str,
    output_dir: Path,
    dpi: int = 300,
    show: bool = False,
) -> tuple[Path, Path]:
    """Plot all model ROC curves for one task."""
    subset = roc_table.loc[roc_table["task"] == task].copy()
    if subset.empty:
        raise ValueError(f"No ROC results found for task: {task}")

    fig, ax = plt.subplots(figsize=(6, 6))

    subset = subset.sort_values("auroc", ascending=False)

    for row in subset.itertuples(index=False):
        ax.plot(
            row.fpr,
            row.tpr,
            linewidth=2,
            label=f"{row.model} (AUROC={row.auroc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{task} ROC")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task = make_safe_name(task)
    png_path = output_dir / f"{safe_task}_roc.png"
    pdf_path = output_dir / f"{safe_task}_roc.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return png_path, pdf_path


def plot_all_tasks(
    roc_table: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
    show: bool = False,
) -> list[tuple[Path, Path]]:
    """Create one ROC figure per task."""
    outputs = []

    for task in dict.fromkeys(roc_table["task"]):
        outputs.append(
            plot_task(
                roc_table=roc_table,
                task=task,
                output_dir=output_dir,
                dpi=dpi,
                show=show,
            )
        )

    return outputs


def save_summary(roc_table: pd.DataFrame, output_dir: Path) -> Path:
    """Save task/model AUROC values to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "roc_summary.csv"

    summary = roc_table[["task", "model", "auroc", "n_samples"]].copy()
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    """Load predictions, compute ROC curves, and save figures."""
    args = parse_args()
    predictions = load_predictions(args.input)
    roc_table = compute_roc_table(predictions)

    outputs = plot_all_tasks(
        roc_table=roc_table,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=args.show,
    )
    summary_path = save_summary(roc_table, args.output_dir)

    for png_path, pdf_path in outputs:
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
