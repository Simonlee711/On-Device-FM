import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Architecture and pretraining configuration reported in the main text (Methods,
# "Model architecture and pretraining objective"). This is the single source of
# truth used by pretrain/himae.py, the downstream scripts, and the demo notebook.
PAPER_CONFIG = {
    "sampling_freq": 100,      # Hz
    "seg_len": 10,             # seconds -> 1,000 samples per window
    "channels": (16, 32, 64, 128, 256),
    "kernel_size": 5,
    "stride": 2,
    "patch_len": 5,            # samples per masking patch
    "mask_ratio": 0.8,
}


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.final_activation = nn.GELU()

    def forward(self, x):
        main_out = self.main_path(x)
        shortcut_out = self.shortcut(x)
        if main_out.size() != shortcut_out.size():
            target_size = main_out.size()[2]
            shortcut_out = shortcut_out[:, :, :target_size]
        return self.final_activation(main_out + shortcut_out)


class DecoderSkipBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride=2, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, output_padding=stride - 1)
        self.conv_block = nn.Sequential(
            nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        if x.size() != skip_connection.size():
            target_size = skip_connection.size()[2]
            x = x[:, :, :target_size]
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv_block(x)


@dataclass
class _CfgView:
    # Minimal config object so the trainer can read patch_len and seq_len
    seq_len: int
    patch_len: int


class HiMAE(nn.Module):
    """Hierarchical 1-D convolutional masked autoencoder (U-Net style).

    - cfg['source'] selects the active modality ('ppg', 'ecg' or 'ppg+ecg').
    - cfg['sampling_freq'] * cfg['seg_len'] gives the window length.
    - cfg['model_params'] may override 'channels', 'kernel_size', 'stride' and
      'patch_len'; anything not given falls back to PAPER_CONFIG.
    - forward(ppg=...) returns {'{mode}_reconstructed': (B,1,L) in [-1,1]}.
    - encode(x) returns the feature map of every encoder stage, i.e. one
      embedding per temporal resolution (see embed_multiresolution).
    """
    def __init__(self, cfg):
        super().__init__()
        self.mode = str(cfg.get('source', 'ppg')).lower()
        if self.mode in ['ppg', 'ecg']:
            in_chans = 1
        elif self.mode == 'ppg+ecg':
            in_chans = 2
        else:
            raise ValueError(f"Unknown source '{self.mode}'")

        seq_len = int(cfg.get('sampling_freq', PAPER_CONFIG['sampling_freq']) * cfg.get('seg_len', PAPER_CONFIG['seg_len']))
        mp = (cfg.get('model_params') or {})
        patch_len = int(mp.get('patch_len', PAPER_CONFIG['patch_len']))
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})")
        channels = tuple(mp.get('channels', PAPER_CONFIG['channels']))
        kernel_size = int(mp.get('kernel_size', PAPER_CONFIG['kernel_size']))
        stride = int(mp.get('stride', PAPER_CONFIG['stride']))

        # Minimal cfg for the trainer
        self.cfg = _CfgView(seq_len=seq_len, patch_len=patch_len)
        self.seq_len = seq_len
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        current_chans = in_chans
        for chan in channels:
            self.encoder_layers.append(ResConvBlock(current_chans, chan, stride=stride, kernel_size=kernel_size))
            current_chans = chan

        reversed_channels = channels[::-1]
        for i in range(len(reversed_channels) - 1):
            in_c, skip_c, out_c = reversed_channels[i], reversed_channels[i+1], reversed_channels[i+1]
            self.decoder_layers.append(DecoderSkipBlock(in_c, skip_c, out_c, stride=stride, kernel_size=kernel_size))

        self.final_deconv = nn.ConvTranspose1d(channels[0], in_chans, kernel_size=kernel_size, stride=stride,
                                               padding=kernel_size // 2, output_padding=stride - 1)
        self.final_activation = nn.Tanh()

    # ---------- Helper to pick the active input and normalize shape ----------
    def _get_active_input(self, ppg, ecg):
        if self.mode == 'ppg+ecg':
            x = torch.cat([ppg.unsqueeze(1), ecg.unsqueeze(1)], dim=1)
            return x
        else:
            x = ppg if self.mode == "ppg" else ecg
            if x is None:
                # When the active modality is not provided (e.g., student pass uses only ppg),
                # silently return empty dict so trainer can skip losses.
                return None
            # Accept (B,L) or (B,1,L); ensure (B,1,L) for convs
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return x

    # ---------- Multi-resolution encoder outputs ----------
    def encode(self, x):
        """Return [h_1, ..., h_K]: the output of every encoder stage.

        h_k has shape (B, channels[k-1], ~L / stride**k): stage 1 is the finest
        temporal resolution (local pulse morphology), stage K the coarsest
        (bottleneck, longer-range rhythm).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = []
        for encoder_block in self.encoder_layers:
            x = encoder_block(x)
            feats.append(x)
        return feats

    def embed_multiresolution(self, x, pooling="mean"):
        """Pool each encoder stage over time -> list of (B, C_k) embeddings.

        pooling: 'mean' (global average pooling, as used for the bottleneck probe),
                 'max', or 'meanmax' (concatenation of both).
        """
        out = []
        for h in self.encode(x):
            if pooling == "mean":
                out.append(h.mean(dim=-1))
            elif pooling == "max":
                out.append(h.amax(dim=-1))
            elif pooling == "meanmax":
                out.append(torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1))
            else:
                raise ValueError(f"Unknown pooling '{pooling}'")
        return out

    def stage_geometry(self):
        """Per encoder stage: channels, downsampling factor, temporal length and receptive field (samples)."""
        k, s = self.kernel_size, self.stride
        rf, jump, length, rows = 1, 1, self.seq_len, []
        for i, c in enumerate(self.channels, start=1):
            # strided conv (k, s) then conv (k, 1); the 1x1 shortcut does not enlarge the field
            rf += (k - 1) * jump
            jump *= s
            rf += (k - 1) * jump
            length = (length + 2 * (k // 2) - k) // s + 1
            rows.append({"stage": i, "channels": c, "downsample": jump, "length": length, "receptive_field": rf})
        return rows

    def reconstruct(self, x):
        """Full encoder-decoder pass on a (B,C,L) tensor; returns (B,C,L) in [-1,1]."""
        skip_connections = self.encode(x)
        bottleneck = skip_connections.pop()
        skip_connections = skip_connections[::-1]

        # Decoder with skips
        current_x = bottleneck
        for i, decoder_block in enumerate(self.decoder_layers):
            current_x = decoder_block(current_x, skip_connections[i])

        x_reconstructed = self.final_deconv(current_x)

        # Safety: ensure final length
        if x_reconstructed.shape[-1] != self.seq_len:
            x_reconstructed = F.interpolate(x_reconstructed, size=self.seq_len, mode='linear', align_corners=False)
        return self.final_activation(x_reconstructed)

    def forward(self,
                ppg=None, ecg=None, *,
                ids_keep_ppg=None, ids_restore_ppg=None,
                ids_keep_ecg=None, ids_restore_ecg=None):
        x = self._get_active_input(ppg, ecg)
        if x is None:
            return {}

        x_reconstructed = self.reconstruct(x)
        if self.mode in {'ppg', 'ecg'}:
            return {f"{self.mode}_reconstructed": x_reconstructed}
        return {
            "ppg_reconstructed": x_reconstructed[:, 0, :],
            "ecg_reconstructed": x_reconstructed[:, 1, :],
        }
