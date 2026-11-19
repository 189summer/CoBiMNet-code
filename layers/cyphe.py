import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def split_window_chains(windows: Sequence[int]) -> list[list[int]]:
    """Group multi-scale window lengths into divisibility-ordered pyramid chains."""
    windows = list(windows)
    n = len(windows)
    assigned = [False] * n
    chains = []

    while not all(assigned):
        chain = []
        last_val = None
        for i in range(n - 1, -1, -1):
            val = windows[i]
            if not assigned[i] or i == 0:
                if not chain:
                    chain.append(val)
                    assigned[i] = True
                    last_val = val
                elif last_val % val == 0:
                    chain.append(val)
                    assigned[i] = True
                    last_val = val
        if chain:
            chain.sort()
            chains.append(chain)
    return chains


class CycleEnhancement(nn.Module):
    """Phase-indexed cycle bank with temporal alignment and bounded gated residual fusion."""

    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        cycle_len: int,
        period_len: int = 24,
        gate_init: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.cycle_len = cycle_len
        self.period_len = period_len

        self.cycle_bank = nn.Parameter(torch.zeros(cycle_len, enc_in))
        self.linear = nn.Linear(seq_len, seq_len)
        half_span = period_len // 2
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, 1 + 2 * half_span),
            stride=1,
            padding=(0, half_span),
            padding_mode="zeros",
            bias=False,
        )
        self.gate = nn.Parameter(torch.tensor(math.atanh(gate_init), dtype=torch.float32))

    def forward(self, x_norm: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        if cycle_index is None:
            raise ValueError("cycle_index is required when cycle enhancement is enabled.")
        B, L, N = x_norm.shape
        x_c = x_norm.permute(0, 2, 1)  # (B, N, L)

        offsets = torch.arange(L, device=cycle_index.device).view(1, -1)
        gather_idx = (cycle_index.view(-1, 1).long() + offsets) % self.cycle_len
        retrieved = self.cycle_bank[gather_idx].permute(0, 2, 1)  # (B, N, L)

        aligned = self.linear(retrieved)
        stacked = torch.stack([x_c, aligned], dim=2).reshape(B * N, 1, 2, L)
        delta = self.conv2d(stacked).reshape(B, N, L).permute(0, 2, 1)
        return x_norm + torch.tanh(self.gate) * delta


class FineScaleAttentionReadout(nn.Module):
    """Self-attention readout with fixed sinusoidal positional encoding for the finest pyramid scale."""

    def __init__(
        self,
        in_channels: int,
        d_head: int,
        num_tokens: int,
        nhead: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_linear = nn.Linear(in_channels, d_head, bias=True)

        position = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_head, 2, dtype=torch.float32) * (-math.log(10000.0) / d_head)
        )
        encoding = torch.zeros(1, num_tokens, d_head)
        encoding[0, :, 0::2] = torch.sin(position * frequencies)
        encoding[0, :, 1::2] = torch.cos(position * frequencies[: encoding[0, :, 1::2].shape[-1]])
        self.register_buffer("positional_encoding", encoding, persistent=True)

        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_head,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_linear(x.permute(0, 2, 1))
        hidden = hidden + self.positional_encoding[:, : hidden.shape[1], :]
        hidden = self.encoder(hidden)
        return hidden[:, -1, :].unsqueeze(0)


class GRUReadout(nn.Module):
    """Single-layer unidirectional GRU returning the final hidden state."""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hn = self.gru(x.permute(0, 2, 1))
        return hn


class ScaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, readout: nn.Module):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.readout = readout

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward_readout(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(x)


class PyramidalHybridEncoder(nn.Module):
    """Dual-chain strided temporal pyramid with heterogeneous fine/coarse readouts."""

    def __init__(
        self,
        seq_len: int,
        windows: Sequence[int] = (24, 48, 72, 144),
        d_model: int = 512,
        conv_channels: int = 128,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.windows = list(windows)
        self.chains = split_window_chains(self.windows)

        counts = {w: 0 for w in self.windows}
        for chain in self.chains:
            for w in chain:
                counts[w] += 1

        base_dim = d_model // len(self.windows)
        self.hid_dims = {w: int(base_dim / counts[w]) for w in self.windows}

        chain_modules = []
        total_concat_dim = 0
        for chain_idx, chain in enumerate(self.chains):
            blocks = []
            prev_scale = 1
            for level_idx, scale in enumerate(chain):
                kernel = int(scale // prev_scale)
                in_ch = 1 if prev_scale == 1 else conv_channels
                hid_dim = self.hid_dims[scale]
                total_concat_dim += hid_dim

                if chain_idx == 0 and level_idx == 0:
                    num_tokens = seq_len // scale
                    eff_heads = next(h for h in range(attn_heads, 0, -1) if hid_dim % h == 0)
                    readout = FineScaleAttentionReadout(
                        in_channels=conv_channels,
                        d_head=hid_dim,
                        num_tokens=num_tokens,
                        nhead=eff_heads,
                        ffn_dim=2 * hid_dim,
                        dropout=attn_dropout,
                    )
                else:
                    readout = GRUReadout(conv_channels, hid_dim)

                blocks.append(ScaleBlock(in_ch, conv_channels, kernel, kernel, readout))
                prev_scale = scale
            chain_modules.append(nn.ModuleList(blocks))

        self.chains_blocks = nn.ModuleList(chain_modules)
        self.out_proj = nn.Linear(total_concat_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * N, 1, L)

        scale_tokens = []
        for chain_idx, blocks in enumerate(self.chains_blocks):
            scales = self.chains[chain_idx]
            conv_features = {}
            feat = x_flat
            for level_idx, block in enumerate(blocks):
                feat = block.forward_conv(feat)
                conv_features[scales[level_idx]] = feat

            last_scale = None
            upsampled = feat
            for level_idx in range(len(scales) - 1, -1, -1):
                block = blocks[level_idx]
                scale = scales[level_idx]
                if last_scale is not None:
                    upsampled = F.interpolate(
                        upsampled,
                        scale_factor=last_scale / scale,
                        mode="nearest",
                    )
                    diff = upsampled.shape[-1] - conv_features[scale].shape[-1]
                    if diff < 0:
                        upsampled = torch.cat(
                            [upsampled, upsampled[:, :, -1:].expand(-1, -1, -diff)],
                            dim=-1,
                        )
                    elif diff > 0:
                        upsampled = upsampled[:, :, : conv_features[scale].shape[-1]]
                    readout_in = upsampled + conv_features[scale]
                else:
                    readout_in = upsampled

                hn = block.forward_readout(readout_in)
                scale_tokens.append(hn.reshape(B, N, self.hid_dims[scale]))
                last_scale = scale

        fused = torch.cat(scale_tokens, dim=-1)
        return self.out_proj(fused)


class CyPHE(nn.Module):
    """Cycle-enhanced Pyramidal Hybrid Encoder."""

    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        cycle_len: int = 24,
        period_len: int = 24,
        gate_init: float = 0.3,
        windows: Sequence[int] = (24, 48, 72, 144),
        d_model: int = 512,
    ):
        super().__init__()
        self.cycle_enhancement = CycleEnhancement(
            seq_len=seq_len,
            enc_in=enc_in,
            cycle_len=cycle_len,
            period_len=period_len,
            gate_init=gate_init,
        )
        self.pyramid = PyramidalHybridEncoder(
            seq_len=seq_len,
            windows=windows,
            d_model=d_model,
        )

    def forward(self, x_norm: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        x_aug = self.cycle_enhancement(x_norm, cycle_index)
        return self.pyramid(x_aug)
