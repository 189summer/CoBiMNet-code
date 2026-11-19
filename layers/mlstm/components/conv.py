# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


class CausalConv1d(nn.Module):
    """Causal depthwise 1D convolution over sequence tensors of shape (B, S, F)."""

    config_class = CausalConv1dConfig

    def __init__(self, config: CausalConv1dConfig):
        super().__init__()
        self.config = config
        self.groups = 1 if config.channel_mixing else config.feature_dim
        if config.kernel_size == 0:
            self.conv = None
        else:
            self.pad = config.kernel_size - 1
            self.conv = nn.Conv1d(
                in_channels=config.feature_dim,
                out_channels=config.feature_dim,
                kernel_size=config.kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=config.causal_conv_bias,
                **config.conv1d_kwargs,
            )
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        if self.conv is not None:
            self.conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        return_last_state: bool = False,
    ) -> torch.Tensor:
        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=1)

        if self.config.kernel_size == 0:
            return x

        y = self.conv(x.transpose(2, 1))
        if conv_state is not None:
            y = y[:, :, conv_state.shape[1] :]

        out = y[:, :, : -self.pad].transpose(2, 1)
        if return_last_state:
            return out, x[:, -self.pad :]
        return out
