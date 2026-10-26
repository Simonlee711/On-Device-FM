#!/usr/bin/env python3
"""Analytic parameter count, model-weight memory and FLOPs for HiMAE.

Counts are computed from the model definition with a single dummy forward pass
(batch size 1, one 10-second window), so no trained weights or device are needed.

Conventions
  * params      : number of parameters (millions).
  * memory_mb   : model-weight memory = params x bytes per parameter (fp32 = 4 bytes),
                  in MB (1e6 bytes); memory_mib gives the same in MiB (2**20 bytes).
                  Buffers (BatchNorm running stats) are excluded.
  * flops_gflops: multiply-accumulates of Conv1d / ConvTranspose1d / Linear layers,
                  one MAC counted as one FLOP (the fvcore convention). Use
                  --flops-per-mac 2 to count a MAC as two FLOPs. Normalization and
                  activation layers are not counted.

Scopes
  * deployed    : encoder + mean-pool + linear head (what runs on-device downstream).
  * autoencoder : full encoder-decoder used for masked pretraining.

Outputs:
    --output         one row per scope: model,scope,params_m,memory_mb,memory_mib,flops_gflops,input_len
    --static-output  one row for --static-scope (default autoencoder, the 1.2M-parameter
                     configuration reported in Table 1), in the schema read by
                     summarize_device_energy.py --static: model,params_m,memory_mb,flops_gflops
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pvc" / "utils" / "model_arch"))
from himae import HiMAE  # noqa: E402


class DeployedHiMAE(nn.Module):
    """Encoder + global average pool + linear head, as used for downstream prediction."""

    def __init__(self, backbone: HiMAE, n_out: int = 1):
        super().__init__()
        self.encoder_layers = backbone.encoder_layers
        self.head = nn.Linear(backbone.encoder_layers[-1].main_path[0].out_channels, n_out)

    def forward(self, x):
        for block in self.encoder_layers:
            x = block(x)
        return self.head(x.mean(dim=-1))


class AutoencoderWrapper(nn.Module):
    def __init__(self, backbone: HiMAE):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(ppg=x)


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    total = 0

    def conv_hook(m, inp, out):
        nonlocal total
        # Each output element needs (C_in / groups) * k MACs.
        total += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0]

    def deconv_hook(m, inp, out):
        nonlocal total
        # Each input element scatters into (C_out / groups) * k outputs.
        x_in = inp[0]
        total += x_in.numel() // x_in.shape[0] * (m.out_channels // m.groups) * m.kernel_size[0]

    def linear_hook(m, inp, out):
        nonlocal total
        total += out.numel() // out.shape[0] * m.in_features

    handles = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv1d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.ConvTranspose1d):
            handles.append(mod.register_forward_hook(deconv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="HiMAE (1.2M)")
    p.add_argument("--sampling-freq", type=int, default=100)
    p.add_argument("--segment-seconds", type=int, default=10)
    p.add_argument("--patch-len", type=int, default=5)
    p.add_argument("--bytes-per-param", type=int, default=4, help="4 = fp32 (full precision).")
    p.add_argument("--flops-per-mac", type=int, default=1, choices=[1, 2])
    p.add_argument("--output", type=Path, default=Path("results/model_complexity.csv"))
    p.add_argument("--static-scope", choices=["deployed", "autoencoder"], default="autoencoder")
    p.add_argument("--static-output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {"source": "ppg", "sampling_freq": args.sampling_freq, "seg_len": args.segment_seconds,
           "model_params": {"patch_len": args.patch_len}}
    backbone = HiMAE(cfg)
    seq_len = args.sampling_freq * args.segment_seconds
    x = torch.randn(1, 1, seq_len)

    rows = []
    for scope, model in (("deployed", DeployedHiMAE(backbone)), ("autoencoder", AutoencoderWrapper(backbone))):
        n_params = sum(p.numel() for p in model.parameters())
        macs = count_macs(model, x)
        rows.append({
            "model": args.model_name, "scope": scope,
            "params_m": n_params / 1e6,
            "memory_mb": n_params * args.bytes_per_param / 1e6,
            "memory_mib": n_params * args.bytes_per_param / 2**20,
            "flops_gflops": macs * args.flops_per_mac / 1e9,
            "input_len": seq_len,
        })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.static_output is not None:
        args.static_output.parent.mkdir(parents=True, exist_ok=True)
        out.loc[out.scope == args.static_scope, ["model", "params_m", "memory_mb", "flops_gflops"]].to_csv(
            args.static_output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
