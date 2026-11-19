import torch
from torch import nn

from .mlstm import mLSTMLayer, mLSTMLayerConfig


class ContextExtractor(nn.Module):
    """Lightweight folded pointwise global context extractor (165 parameters at inplanes=8, ratio=0.5)."""

    def __init__(self, d_model: int, inplanes: int = 8, ratio: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.inplanes = inplanes
        self.planes = int(inplanes * ratio)

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, inplanes, kernel_size=1),
        )
        self.out_conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)

    def _pool(self, u: torch.Tensor) -> torch.Tensor:
        B, C, N, G = u.shape
        values = u.view(B, C, N * G).unsqueeze(1)
        weights = self.softmax(self.conv_mask(u).view(B, 1, N * G)).unsqueeze(-1)
        return torch.matmul(values, weights).view(B, C, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        u = x.reshape(B, self.inplanes, N, D // self.inplanes)
        ctx = self._pool(u)
        delta = self.out_conv(self.channel_mlp(ctx))
        return (u + delta).reshape(B, N, D)


class CoBiM(nn.Module):
    """Context-Guided Bidirectional Variable Memory module."""

    def __init__(
        self,
        d_model: int = 512,
        context_length: int = 512,
        num_heads: int = 4,
        conv1d_kernel_size: int = 8,
        qkv_proj_blocksize: int = 4,
        proj_factor: float = 1.0,
        dropout: float = 0.4,
        bidirectional: bool = True,
        use_context: bool = True,
    ):
        super().__init__()
        self.memory = mLSTMLayer(
            mLSTMLayerConfig(
                embedding_dim=d_model,
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=qkv_proj_blocksize,
                num_heads=num_heads,
                proj_factor=proj_factor,
                bias=True,
                dropout=dropout,
                context_length=context_length,
                variable_mask_mode="bidirectional" if bidirectional else "causal",
            )
        )
        self.context = ContextExtractor(d_model, inplanes=8, ratio=0.5) if use_context else None
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mem = self.memory(h)
        if self.context is not None:
            u = mem * self.context(h) + h
        else:
            u = mem + h
        return self.dropout(self.layer_norm(u) + h)
