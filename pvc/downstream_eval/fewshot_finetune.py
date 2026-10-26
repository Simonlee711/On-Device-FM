"""Generate long-form few-shot AUROC CSVs consumed by ``plot/fig4.py``.

Example (synthetic PVC files shipped with this repository)::

    python pvc/downstream_eval/fewshot_adaptation.py \
        --h5 pvc_10s_synth.h5 \
        --checkpoint himae_synth.ckpt \
        --dataset PVC \
        --model-name "HiMAE (1.2M)" \
        --shots 256 512 1024 2048 all \
        --runs 3 \
        --include-frozen \
        --output results/finetuning_himae.csv

The resulting file can be passed directly to::

    python plot/fig4.py --input results/finetuning_himae.csv --output-dir figures
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

# Allow execution from anywhere inside/outside the repository.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvc.utils.model_arch.himae import HiMAE  # noqa: E402


OUTPUT_COLUMNS = ["dataset", "model", "shots", "run", "auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen linear-probe and few-shot fine-tuning experiments."
    )
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "finetuning.csv")
    parser.add_argument("--dataset", default="PVC")
    parser.add_argument("--model-name", default="HiMAE (1.2M)")
    parser.add_argument("--signal-key", default="ppg")
    parser.add_argument("--label-key", default="labels")
    parser.add_argument("--patient-key", default="patient_ids")
    parser.add_argument("--sampling-freq", type=int, default=25)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--patch-len", type=int, default=50)
    parser.add_argument(
        "--shots",
        nargs="+",
        default=["256", "512", "1024", "2048", "all"],
        help='Fine-tuning sample counts. Use integers and/or "all".',
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--include-frozen",
        action="store_true",
        help="Also emit shots=0 frozen-encoder linear-probe runs.",
    )
    parser.add_argument("--split-seed", type=int, default=77)
    parser.add_argument("--run-seed", type=int, default=1000)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV and de-duplicate dataset/model/shots/run rows.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def minmax_per_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    scale = np.maximum(hi - lo, 1e-8)
    return 2.0 * (x - lo) / scale - 1.0


def load_arrays(
    path: Path, signal_key: str, label_key: str, patient_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        x = np.asarray(h5[signal_key][:], dtype=np.float32)
        y = np.asarray(h5[label_key][:]).reshape(-1).astype(np.int64)
        if patient_key in h5:
            patient_ids = np.asarray(h5[patient_key][:]).reshape(-1)
        else:
            patient_ids = np.arange(len(y))

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(f"Expected signals [N,L] or [N,1,L], got {x.shape}")
    if not (len(x) == len(y) == len(patient_ids)):
        raise ValueError("Signal, label, and participant arrays have different lengths.")
    if np.unique(y).size != 2:
        raise ValueError(f"This runner expects binary labels; found {np.unique(y)}")

    return minmax_per_window(x), y, patient_ids


def participant_disjoint_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed participant-disjoint train/val/test indices."""
    all_idx = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_rel, test_rel = next(outer.split(all_idx, y, groups=groups))

    train_val_idx = all_idx[train_val_rel]
    test_idx = all_idx[test_rel]
    train_val_groups = groups[train_val_idx]
    train_val_y = y[train_val_idx]

    # val_fraction is interpreted as a fraction of the non-test pool.
    inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(train_val_idx, train_val_y, groups=train_val_groups)
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        if np.unique(y[idx]).size != 2:
            raise RuntimeError(
                f"{name} split contains only one class. Change --split-seed or split fractions."
            )

    return train_idx, val_idx, test_idx


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[item]
        return torch.from_numpy(self.x[idx]).float(), torch.tensor(self.y[idx]).float()


class EncoderClassifier(nn.Module):
    def __init__(self, backbone: HiMAE):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(256, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        current = x.unsqueeze(1) if x.ndim == 2 else x
        for block in self.backbone.encoder_layers:
            current = block(current)
        return current.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x)).squeeze(-1)


def build_backbone(args: argparse.Namespace, device: torch.device) -> HiMAE:
    cfg = {
        "source": args.signal_key,
        "sampling_freq": args.sampling_freq,
        "seg_len": args.segment_seconds,
        "model_params": {"patch_len": args.patch_len},
    }
    backbone = HiMAE(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("model.", "backbone.", "net.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        cleaned[new_key] = value

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    encoder_missing = [k for k in missing if k.startswith("encoder_layers")]
    if encoder_missing:
        raise RuntimeError(f"Checkpoint is missing encoder weights: {encoder_missing[:5]}")
    if unexpected:
        print(f"Warning: ignored {len(unexpected)} unexpected checkpoint keys.")
    return backbone.to(device)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ArrayDataset(x, y, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_auc(model: EncoderClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ys, scores = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        ys.append(yb.numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return float(roc_auc_score(np.concatenate(ys), np.concatenate(scores)))


def train_model(
    model: EncoderClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    freeze_encoder: bool,
) -> EncoderClassifier:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(not freeze_encoder)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    best_state = None
    stale = 0

    for _epoch in range(epochs):
        model.train()
        if freeze_encoder:
            model.backbone.eval()  # keep BatchNorm frozen for a true linear probe
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_auc = evaluate_auc(model, val_loader, device)
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model


def stratified_subsample(indices: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(indices):
        return np.asarray(indices)
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    pos = indices[y[indices] == 1]
    neg = indices[y[indices] == 0]

    # Preserve the training prevalence while guaranteeing both classes whenever n >= 2.
    n_pos = int(round(n * len(pos) / len(indices)))
    n_pos = min(max(n_pos, 1), len(pos), n - 1)
    n_neg = n - n_pos
    if n_neg > len(neg):
        n_neg = len(neg)
        n_pos = n - n_neg
    chosen = np.concatenate(
        [rng.choice(pos, size=n_pos, replace=False), rng.choice(neg, size=n_neg, replace=False)]
    )
    rng.shuffle(chosen)
    return chosen


def normalize_shots(values: Iterable[str]) -> list[str]:
    normalized = []
    for value in values:
        text = str(value).strip().lower()
        if text == "all":
            normalized.append("all")
        else:
            n = int(text)
            if n <= 0:
                raise ValueError("Positive fine-tuning shot counts are required; use --include-frozen for shots=0.")
            normalized.append(str(n))
    return list(dict.fromkeys(normalized))


def row(dataset: str, model: str, shots: str, run: int, auc: float) -> dict[str, object]:
    return {"dataset": dataset, "model": model, "shots": shots, "run": run, "auc": auc}


def save_rows(rows: list[dict[str, object]], path: Path, append: bool) -> None:
    new_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if append and path.exists():
        old_df = pd.read_csv(path)
        missing = set(OUTPUT_COLUMNS).difference(old_df.columns)
        if missing:
            raise ValueError(f"Existing output is missing columns: {sorted(missing)}")
        new_df = pd.concat([old_df[OUTPUT_COLUMNS], new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(
            subset=["dataset", "model", "shots", "run"], keep="last"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    shots = normalize_shots(args.shots)
    print(f"Device: {device}")

    x, y, patient_ids = load_arrays(
        args.h5, args.signal_key, args.label_key, args.patient_key
    )
    train_idx, val_idx, test_idx = participant_disjoint_split(
        y,
        patient_ids,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )
    print(
        f"Split windows: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}; "
        f"participants={len(np.unique(patient_ids))}"
    )

    val_loader = make_loader(x, y, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(x, y, test_idx, args.batch_size, shuffle=False)
    rows: list[dict[str, object]] = []

    for run in range(args.runs):
        seed = args.run_seed + run
        set_seed(seed)

        if args.include_frozen:
            backbone = build_backbone(args, device)
            model = EncoderClassifier(backbone).to(device)
            frozen_train_loader = make_loader(
                x, y, train_idx, args.batch_size, shuffle=True
            )
            model = train_model(
                model,
                frozen_train_loader,
                val_loader,
                device,
                lr=args.probe_lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                freeze_encoder=True,
            )
            auc = evaluate_auc(model, test_loader, device)
            rows.append(row(args.dataset, args.model_name, "0", run, auc))
            print(f"run={run} shots=0 test_auc={auc:.4f}")

        for shot in shots:
            set_seed(seed)
            subset = (
                train_idx
                if shot == "all"
                else stratified_subsample(train_idx, y, int(shot), seed)
            )
            train_loader = make_loader(x, y, subset, args.batch_size, shuffle=True)
            backbone = build_backbone(args, device)
            model = EncoderClassifier(backbone).to(device)
            model = train_model(
                model,
                train_loader,
                val_loader,
                device,
                lr=args.finetune_lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                freeze_encoder=False,
            )
            auc = evaluate_auc(model, test_loader, device)
            rows.append(row(args.dataset, args.model_name, shot, run, auc))
            print(f"run={run} shots={shot} n_train={len(subset)} test_auc={auc:.4f}")

    save_rows(rows, args.output, args.append)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
