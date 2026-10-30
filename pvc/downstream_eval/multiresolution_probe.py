#!/usr/bin/env python3
"""Resolution-specific linear probes on frozen HiMAE encoder stages.

HiMAE's encoder produces one feature map per stage, each at a coarser temporal
resolution (stride 2 per stage). This script tests the resolution hypothesis:
which temporal resolution carries the most task-relevant information.

For every encoder stage k (and the concatenation of all stages) it
  1. extracts the stage-k feature map from the frozen pretrained encoder and
     pools it over time (mean by default) -> one embedding per window;
  2. fits an L2-regularized logistic-regression probe on the training split,
     choosing the regularization strength on the validation split;
  3. reports validation and held-out test AUROC with participant-level bootstrap
     95% CIs, and the paired test AUROC difference against the final bottleneck
     (stage K, the embedding used by the standard linear probe);
  4. selects the best stage on the validation split only and reports its test AUROC.

Splits are participant-disjoint (train / val / test). The prediction export uses the
long-form schema of analysis/, so operating-point analyses can be run per stage.

Example (synthetic PVC files shipped with this repository; 25 Hz input is
resampled to the 100 Hz pretraining rate):

    python pvc/downstream_eval/multiresolution_probe.py \
        --h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt --task PVC \
        --output-dir results/multiresolution

    python plot/fig_multiresolution.py \
        --input results/multiresolution/multires_probe_summary.csv --output-dir figures

Outputs (in --output-dir):
    multires_probe_summary.csv   task,level,stage,channels,downsample,temporal_length,
                                 receptive_field_samples,receptive_field_ms,time_step_ms,
                                 n_features,C,val_auroc,test_auroc,ci_low,ci_high,
                                 selected_on_validation,...
    multires_probe_paired.csv    task,level,reference,delta_auroc,ci_low,ci_high,p_value
    multires_predictions.csv     task,model,regime,split,person_id,sample_id,y_true,y_score
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pvc.utils.model_arch.himae import HiMAE, PAPER_CONFIG  # noqa: E402
from pvc.utils.signal_utils import load_h5_windows, participant_disjoint_split  # noqa: E402
from _bootstrap import bootstrap_indices, paired_p_value, percentile_ci  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="PVC")
    p.add_argument("--model-name", default="HiMAE (1.2M)")
    p.add_argument("--signal-key", default="ppg")
    p.add_argument("--label-key", default="labels")
    p.add_argument("--patient-key", default="patient_ids")
    p.add_argument("--input-fs", type=int, default=None,
                   help="Sampling rate of the stored windows (inferred from length if omitted).")
    p.add_argument("--pooling", choices=["mean", "max", "meanmax"], default="mean")
    p.add_argument("--C-grid", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    p.add_argument("--split-seed", type=int, default=77)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024, help="Bootstrap seed.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "multiresolution")
    return p.parse_args()


def load_backbone(checkpoint: Path) -> HiMAE:
    backbone = HiMAE({"source": "ppg", "sampling_freq": PAPER_CONFIG["sampling_freq"],
                      "seg_len": PAPER_CONFIG["seg_len"], "model_params": {}})
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    cleaned = {}
    for k, v in sd.items():
        for prefix in ("model.", "backbone.", "net.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    missing, _ = backbone.load_state_dict(cleaned, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoder_layers")]
    if enc_missing:
        raise RuntimeError(f"Checkpoint is missing encoder weights: {enc_missing[:5]}")
    return backbone.eval()


@torch.no_grad()
def extract(backbone: HiMAE, x: np.ndarray, pooling: str, batch_size: int, device) -> list[np.ndarray]:
    backbone = backbone.to(device)
    chunks: list[list[np.ndarray]] = [[] for _ in backbone.encoder_layers]
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).unsqueeze(1).to(device)
        for k, e in enumerate(backbone.embed_multiresolution(xb, pooling=pooling)):
            chunks[k].append(e.cpu().numpy())
    return [np.concatenate(c) for c in chunks]


def fit_probe(z, y, tr, va, c_grid):
    best = None
    for c in c_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        clf.fit(z[tr], y[tr])
        auc = roc_auc_score(y[va], clf.predict_proba(z[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    return best


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if np.unique(y).size == 2 else float("nan")


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    x, y, pid = load_h5_windows(args.h5, args.signal_key, args.label_key, args.patient_key,
                                input_fs=args.input_fs, model_fs=PAPER_CONFIG["sampling_freq"],
                                seg_len=PAPER_CONFIG["seg_len"])
    tr, va, te = participant_disjoint_split(y, pid, args.test_fraction, args.val_fraction, args.split_seed)
    print(f"windows: train={len(tr)} val={len(va)} test={len(te)}; participants={len(np.unique(pid))}")

    backbone = load_backbone(args.checkpoint)
    geometry = backbone.stage_geometry()
    feats = extract(backbone, x, args.pooling, args.batch_size, device)
    n_stages = len(feats)
    levels = [(f"stage{g['stage']}", feats[i], g) for i, g in enumerate(geometry)]
    levels.append(("all_stages", np.concatenate(feats, axis=1), None))
    ref_level = f"stage{n_stages}"
    fs = PAPER_CONFIG["sampling_freq"]

    yte, pte = y[te], pid[te]
    boot_idx = list(bootstrap_indices(pte, args.n_boot, args.seed))
    sample_id = pd.Series(np.arange(len(y))).groupby(pid).cumcount().to_numpy()

    rows, preds, boot_auc = [], [], {}
    for name, z, g in levels:
        val_auc, c, clf = fit_probe(z, y, tr, va, args.C_grid)
        s_va, s_te = clf.predict_proba(z[va])[:, 1], clf.predict_proba(z[te])[:, 1]
        boot_auc[name] = np.array([safe_auc(yte[i], s_te[i]) for i in boot_idx])
        lo, hi = percentile_ci(boot_auc[name])
        rows.append({
            "task": args.task, "level": name,
            "stage": g["stage"] if g else "all",
            "channels": g["channels"] if g else z.shape[1],
            "downsample": g["downsample"] if g else np.nan,
            "temporal_length": g["length"] if g else np.nan,
            "receptive_field_samples": g["receptive_field"] if g else np.nan,
            "receptive_field_ms": 1000.0 * g["receptive_field"] / fs if g else np.nan,
            "time_step_ms": 1000.0 * g["downsample"] / fs if g else np.nan,
            "n_features": z.shape[1], "C": c, "pooling": args.pooling,
            "val_auroc": val_auc, "test_auroc": safe_auc(yte, s_te), "ci_low": lo, "ci_high": hi,
            "n_test": len(te), "n_test_participants": len(np.unique(pte)),
            "n_boot": args.n_boot, "seed": args.seed, "split_seed": args.split_seed,
        })
        for split, idx, s in (("val", va, s_va), ("test", te, s_te)):
            preds.append(pd.DataFrame({
                "task": args.task, "model": args.model_name, "regime": f"Frozen ({name})", "split": split,
                "person_id": pid[idx], "sample_id": sample_id[idx], "y_true": y[idx], "y_score": s,
            }))
        print(f"{name:>10}: val AUROC={val_auc:.4f}  test AUROC={rows[-1]['test_auroc']:.4f} [{lo:.4f}, {hi:.4f}]")

    summary = pd.DataFrame(rows)
    single = summary[summary.level != "all_stages"]
    best = single.loc[single.val_auroc.idxmax(), "level"]
    summary["selected_on_validation"] = summary.level.eq(best)
    print(f"validation-selected stage: {best}")

    paired = []
    for r in rows:
        if r["level"] == ref_level:
            continue
        d = boot_auc[r["level"]] - boot_auc[ref_level]
        lo, hi = percentile_ci(d)
        ref_auc = summary.loc[summary.level == ref_level, "test_auroc"].item()
        paired.append({"task": args.task, "level": r["level"], "reference": ref_level,
                       "delta_auroc": r["test_auroc"] - ref_auc, "ci_low": lo, "ci_high": hi,
                       "p_value": paired_p_value(d), "n_boot": args.n_boot, "seed": args.seed})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "multires_probe_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output_dir / "multires_probe_paired.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(args.output_dir / "multires_predictions.csv", index=False)
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
