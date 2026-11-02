#!/usr/bin/env python3
"""Frozen-encoder linear probe for binary window classification (script form of the demo notebook).

This replaces the earlier research-environment script of the same name, which
depended on an unreleased task registry (downstream_eval.task_definition), S3
paths and window-level (participant-overlapping) cross-validation. It runs the
same procedure as HiMAE_PVC_Detection.ipynb ("Linear probe on the bottleneck"):

  1. load windows, resample to the 100 Hz pretraining rate and min-max normalize
     each window (pvc/utils/signal_utils.py);
  2. participant-disjoint train / val / test split (--split-seed, identical to
     fewshot_finetune.py and multiresolution_probe.py);
  3. frozen HiMAE encoder, mean-pooled embedding of --embedding-stage
     (default: the stage-5 bottleneck);
  4. one logistic layer trained with AdamW on standardized embeddings, the epoch
     with the best validation AUROC kept; the test split is used once, at the end;
  5. export validation and test predictions in the canonical schema
     (analysis/prediction_schema.py), ready for analysis/run_statistical_workflow.sh.

Example (synthetic PVC files in this repository):

    python pvc/downstream_eval/binary_linear_prob.py \
        --h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt \
        --task PVC --model-name "HiMAE (1.2M)" --regime Frozen \
        --predictions-output results/pvc_predictions.csv \
        --probe-output results/pvc_linear_probe.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "analysis", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pvc.utils.model_arch.himae import PAPER_CONFIG  # noqa: E402
from pvc.utils.signal_utils import load_h5_windows, participant_disjoint_split  # noqa: E402
from multiresolution_probe import extract, load_backbone  # noqa: E402
from prediction_schema import make_predictions_frame, window_ids_within_participant  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="PVC")
    p.add_argument("--model-name", default="HiMAE (1.2M)")
    p.add_argument("--regime", default="Frozen")
    p.add_argument("--embedding-stage", default="last",
                   help='Encoder stage to probe: "last" (bottleneck), 1-5, or "all" (concatenation).')
    p.add_argument("--signal-key", default="ppg")
    p.add_argument("--label-key", default="labels")
    p.add_argument("--patient-key", default="patient_ids")
    p.add_argument("--input-fs", type=int, default=None,
                   help="Sampling rate of the stored windows (inferred from length if omitted).")
    p.add_argument("--seed", type=int, default=7, help="Probe initialization / shuffling seed.")
    p.add_argument("--split-seed", type=int, default=77)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--extract-batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--predictions-output", type=Path, default=REPO_ROOT / "results" / "pvc_predictions.csv")
    p.add_argument("--probe-output", type=Path, default=None,
                   help="Optional path for the probe weights (acting on raw pooled embeddings).")
    return p.parse_args(argv)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats):
        if feats.dim() == 3:          # (B, C, T') feature map -> global average pooling
            feats = feats.mean(dim=-1)
        return self.fc(feats).squeeze(-1)


def train_linear_probe(z, y, tr, va, device, seed, epochs, lr, weight_decay, batch_size):
    """Train on tr, keep the epoch with the best AUROC on va; standardization is folded into the layer."""
    torch.manual_seed(seed)
    mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-6
    zt = torch.from_numpy((z - mu) / sd).float()
    probe = LinearProbe(z.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    yt = torch.from_numpy(y).float()
    best, best_state = -np.inf, None
    for _ in range(epochs):
        probe.train()
        for b in np.array_split(np.random.permutation(tr), max(1, len(tr) // batch_size)):
            loss = nn.functional.binary_cross_entropy_with_logits(probe(zt[b].to(device)), yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            val_auc = roc_auc_score(y[va], torch.sigmoid(probe(zt[va].to(device))).cpu().numpy())
        if val_auc > best:
            best, best_state = val_auc, {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    with torch.no_grad():
        w = probe.fc.weight / torch.from_numpy(sd).float().to(device)
        probe.fc.bias -= (w * torch.from_numpy(mu).float().to(device)).sum(1)
        probe.fc.weight.copy_(w)
    return probe, best


def stage_embedding(feats: list[np.ndarray], stage: str) -> np.ndarray:
    if stage == "last":
        return feats[-1]
    if stage == "all":
        return np.concatenate(feats, axis=1)
    k = int(stage)
    if not 1 <= k <= len(feats):
        raise ValueError(f"--embedding-stage must be 1..{len(feats)}, 'last' or 'all'")
    return feats[k - 1]


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, pid = load_h5_windows(args.h5, args.signal_key, args.label_key, args.patient_key,
                                input_fs=args.input_fs, model_fs=PAPER_CONFIG["sampling_freq"],
                                seg_len=PAPER_CONFIG["seg_len"])
    tr, va, te = participant_disjoint_split(y, pid, args.test_fraction, args.val_fraction, args.split_seed)
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        print(f"{name:>5}: {len(idx):6,d} windows, {len(np.unique(pid[idx])):3d} participants, {int(y[idx].sum()):5,d} positive")

    backbone = load_backbone(args.checkpoint).to(device)
    z = stage_embedding(extract(backbone, x, "mean", args.extract_batch_size, device), args.embedding_stage)
    probe, best_val = train_linear_probe(z, y, tr, va, device, args.seed, args.epochs, args.lr,
                                         args.weight_decay, args.batch_size)
    with torch.no_grad():
        scores = torch.sigmoid(probe(torch.from_numpy(z).float().to(device))).cpu().numpy()
    print(f"validation AUROC={best_val:.4f}  test AUROC={roc_auc_score(y[te], scores[te]):.4f}")

    sample_id = window_ids_within_participant(pid)
    rows = np.concatenate([va, te])
    pred = make_predictions_frame(args.task, args.model_name, args.regime,
                                  np.where(np.isin(rows, te), "test", "val"),
                                  pid[rows], sample_id[rows], y[rows], scores[rows])
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.predictions_output, index=False)
    print(f"Wrote {len(pred):,} canonical prediction rows to {args.predictions_output}")
    if args.probe_output is not None:
        args.probe_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in probe.state_dict().items()}, args.probe_output)
        print(f"Saved probe to {args.probe_output}")


if __name__ == "__main__":
    main()
