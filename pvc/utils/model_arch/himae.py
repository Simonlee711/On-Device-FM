"""HiMAE: a hierarchical 1D convolutional masked autoencoder for physiological signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class ResConvBlock(nn.Module):
    """Residual 1D conv block (two k=5 convs) that downsamples by ``stride``."""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        padding = 2
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=padding, bias=False),
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
    """Upsamples by 2x, concatenates the encoder skip connection, and refines with two convs."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv_block = nn.Sequential(
            nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
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


def _best_divisor_close_to(L: int, preferred: int = 40) -> int:
    """Return the divisor of ``L`` in [8, 256] closest to ``preferred``."""
    if L <= 0:
        return preferred
    cands = [d for d in range(1, L+1) if L % d == 0 and 8 <= d <= min(256, L)]
    if not cands:
        return max(8, min(preferred, L))
    return min(cands, key=lambda d: abs(d - preferred))

@dataclass
class _CfgView:
    """Minimal config view exposing ``seq_len`` and ``patch_len`` to the trainer."""

    seq_len: int
    patch_len: int

class HiMAE(nn.Module):
    """Hierarchical convolutional masked autoencoder.

    - Uses cfg['source'] to select which modality is active ('ppg' or 'ecg').
    - Exposes self.cfg with .patch_len and .seq_len for the trainer.
    - Returns a dict with key '{mode}_reconstructed': (B,1,L) in [-1,1].
    """
    def __init__(self, cfg):
        super().__init__()
        self.mode = str(cfg['source']).lower()
        if self.mode in ['ppg', 'ecg']:
            in_chans = 1
        elif self.mode == 'ppg+ecg':
            in_chans = 2

        seq_len = int(cfg['sampling_freq'] * cfg['seg_len'])
        mp = (cfg.get('model_params') or {})
        patch_len = int(mp.get('patch_len', 0)) or _best_divisor_close_to(seq_len, 40)
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"

        self.cfg = _CfgView(seq_len=seq_len, patch_len=patch_len)
        self.seq_len = seq_len

        # Five-stage encoder; each stage halves the temporal resolution.
        channels = [16, 32, 64, 128, 256]
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        current_chans = in_chans
        for chan in channels:
            self.encoder_layers.append(ResConvBlock(current_chans, chan, stride=2))
            current_chans = chan

        reversed_channels = channels[::-1]
        for i in range(len(reversed_channels) - 1):
            in_c, skip_c, out_c = reversed_channels[i], reversed_channels[i+1], reversed_channels[i+1]
            self.decoder_layers.append(DecoderSkipBlock(in_c, skip_c, out_c))

        self.final_deconv = nn.ConvTranspose1d(channels[0], in_chans, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.final_activation = nn.Tanh()

    def _get_active_input(self, ppg, ecg):
        """Select the input for the configured modality and reshape it to (B, C, L)."""
        if self.mode == 'ppg+ecg':
            x = torch.cat([ppg.unsqueeze(1), ecg.unsqueeze(1)], dim=1)
            return x
        else:
            x = ppg if self.mode == "ppg" else ecg
            if x is None:
                # The configured modality is absent; forward() returns an empty dict.
                return None
            # Accept (B, L) or (B, 1, L).
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return x

    def forward(self,
                ppg=None, ecg=None, *,
                ids_keep_ppg=None, ids_restore_ppg=None,
                ids_keep_ecg=None, ids_restore_ecg=None):
        x = self._get_active_input(ppg, ecg)
        if x is None:
            return {}

        # Encoder
        skip_connections = []
        current_x = x
        for encoder_block in self.encoder_layers:
            current_x = encoder_block(current_x)
            skip_connections.append(current_x)

        bottleneck = skip_connections.pop()
        skip_connections = skip_connections[::-1]

        # Decoder with skips
        current_x = bottleneck
        for i, decoder_block in enumerate(self.decoder_layers):
            skip = skip_connections[i]
            current_x = decoder_block(current_x, skip)

        x_reconstructed = self.final_deconv(current_x)

        # Match the input length exactly if transposed convs over/undershoot.
        if x_reconstructed.shape[-1] != self.seq_len:
            x_reconstructed = F.interpolate(x_reconstructed, size=self.seq_len, mode='linear', align_corners=False)
        if self.mode in {'ppg', 'ecg'}:
            return {
                f"{self.mode}_reconstructed": self.final_activation(x_reconstructed),
            }
        else:
            return {
                "ppg_reconstructed": self.final_activation(x_reconstructed[:, 0, :]),
                "ecg_reconstructed": self.final_activation(x_reconstructed[:, 1, :]),
            }




