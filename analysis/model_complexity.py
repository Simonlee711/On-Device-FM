#!/usr/bin/env python3
"""Analytic parameter count and model-weight memory of the HiMAE model definitions.

Counts are computed from the PyTorch model definition; no trained weights, device
or device log is used, so nothing here is an on-device measurement.

Three different HiMAE artifacts exist; only the first two are in this repository:
  * pretraining_autoencoder : full encoder-decoder trained with masked reconstruction
                              (pretrain/himae.py; weights in himae_synth.ckpt).
  * downstream_classifier   : pretrained encoder + temporal mean-pool + linear head,
                              the model used for frozen linear-probe and fine-tuned
                              evaluation (the decoder is discarded).
  * deployed inference artifact : the TorchScript model executed by the Wear OS
                              benchmarking application on the watch. It was produced
                              by the proprietary deployment toolchain and is not
                              released, so it cannot be counted here.

Conventions
  * params_m     : number of parameters (millions).
  * memory_mb    : model-weight memory = params x bytes per parameter (fp32 = 4 bytes),
                   in MB (1e6 bytes); memory_mib gives the same in MiB (2**20 bytes).
                   Buffers (BatchNorm running stats) are excluded.

Outputs:
    --output         one row per scope: model,scope,params_m,memory_mb,memory_mib
    --static-output  one row for --static-scope in the schema read by
                     summarize_device_benchmarks.py --static: model,scope,params_m,memory_mb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pvc" / "utils" / "model_arch"))
from himae import HiMAE  # noqa: E402


class PretrainingAutoencoder(nn.Module):
    """Full masked autoencoder (encoder + decoder) as pretrained."""

    def __init__(self, backbone: HiMAE):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(ppg=x)


class DownstreamClassifier(nn.Module):
    """Pretrained encoder + temporal mean-pool + linear head (frozen / fine-tuned evaluation model)."""

    def __init__(self, backbone: HiMAE, n_out: int = 1):
        super().__init__()
        self.encoder_layers = backbone.encoder_layers
        self.head = nn.Linear(backbone.encoder_layers[-1].main_path[0].out_channels, n_out)

    def forward(self, x):
        for block in self.encoder_layers:
            x = block(x)
        return self.head(x.mean(dim=-1))


SCOPES = {"pretraining_autoencoder": PretrainingAutoencoder, "downstream_classifier": DownstreamClassifier}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="HiMAE (1.2M)")
    p.add_argument("--sampling-freq", type=int, default=100)
    p.add_argument("--segment-seconds", type=int, default=10)
    p.add_argument("--patch-len", type=int, default=5)
    p.add_argument("--bytes-per-param", type=int, default=4, help="4 = fp32 (full precision).")
    p.add_argument("--output", type=Path, default=Path("results/model_complexity.csv"))
    p.add_argument("--static-scope", choices=list(SCOPES), default="pretraining_autoencoder",
                   help="Scope written to --static-output (Table 1 lists the 1.2M-parameter autoencoder).")
    p.add_argument("--static-output", type=Path, default=None)
    return p.parse_args(argv)


def complexity_table(args) -> pd.DataFrame:
    cfg = {"source": "ppg", "sampling_freq": args.sampling_freq, "seg_len": args.segment_seconds,
           "model_params": {"patch_len": args.patch_len}}
    backbone = HiMAE(cfg)
    rows = []
    for scope, cls in SCOPES.items():
        model = cls(backbone)
        n_params = sum(p.numel() for p in model.parameters())
        rows.append({
            "model": args.model_name, "scope": scope,
            "params_m": n_params / 1e6,
            "memory_mb": n_params * args.bytes_per_param / 1e6,
            "memory_mib": n_params * args.bytes_per_param / 2**20,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    out = complexity_table(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.static_output is not None:
        args.static_output.parent.mkdir(parents=True, exist_ok=True)
        out.loc[out.scope == args.static_scope, ["model", "scope", "params_m", "memory_mb"]].to_csv(
            args.static_output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
