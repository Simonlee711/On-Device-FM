"""Window loading, resampling, normalization and participant-disjoint splitting.

All downstream entry points (demo notebook, linear / multi-resolution probes,
few-shot fine-tuning) use these helpers so that every HiMAE forward pass sees the
input format the released checkpoint was pretrained on (PAPER_CONFIG): 10-second
windows at 100 Hz (1,000 samples), min-max normalized per window to [-1, 1].
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit


def minmax_per_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    scale = np.maximum(hi - lo, 1e-8)
    return 2.0 * (x - lo) / scale - 1.0


def resample_windows(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resampling of every window (rows) from fs_in to fs_out Hz."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float32)
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=1).astype(np.float32)


def load_h5_windows(
    path: Path,
    signal_key: str = "ppg",
    label_key: str = "labels",
    patient_key: str = "patient_ids",
    input_fs: int | None = None,
    model_fs: int = 100,
    seg_len: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (x, y, participant_id); resample to model_fs and min-max normalize.

    If input_fs is None it is inferred from the window length and seg_len.
    """
    with h5py.File(path, "r") as h5:
        x = np.asarray(h5[signal_key][:], dtype=np.float32)
        y = np.asarray(h5[label_key][:]).reshape(-1).astype(np.int64)
        pid = np.asarray(h5[patient_key][:]).reshape(-1) if patient_key in h5 else np.arange(len(y))
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(f"Expected signals [N,L] or [N,1,L], got {x.shape}")
    if not (len(x) == len(y) == len(pid)):
        raise ValueError("Signal, label, and participant arrays have different lengths.")
    if input_fs is None:
        if x.shape[1] % seg_len:
            raise ValueError(f"Cannot infer sampling rate: window length {x.shape[1]} is not a multiple of {seg_len} s.")
        input_fs = x.shape[1] // seg_len
    x = resample_windows(x, input_fs, model_fs)
    if x.shape[1] != model_fs * seg_len:
        raise ValueError(f"Expected {model_fs * seg_len} samples after resampling, got {x.shape[1]}.")
    return minmax_per_window(x), y, pid


def participant_disjoint_split(
    y: np.ndarray, groups: np.ndarray, test_fraction: float, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed participant-disjoint train/val/test indices (val_fraction is of the non-test pool)."""
    all_idx = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_rel, test_rel = next(outer.split(all_idx, y, groups=groups))
    train_val_idx, test_idx = all_idx[train_val_rel], all_idx[test_rel]
    inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(train_val_idx, y[train_val_idx], groups=groups[train_val_idx]))
    train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        if np.unique(y[idx]).size != 2:
            raise RuntimeError(f"{name} split contains only one class. Change the split seed or fractions.")
    return train_idx, val_idx, test_idx
