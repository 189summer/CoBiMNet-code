# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn


@dataclass
class LinearHeadwiseExpandConfig:
    in_features: int = 0
    num_heads: int = -1
    expand_factor_up: float = 1.0
    _out_features: int = -1
    bias: bool = True
    trainable_weight: bool = True
    trainable_bias: bool = True

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be set"
        assert self.num_heads <= self.in_features, "num_heads must be <= in_features"
        assert self.in_features % self.num_heads == 0, "in_features must be divisible by num_heads"
        if self._out_features < 0:
            self._out_features = round(self.expand_factor_up * self.in_features)


class LinearHeadwiseExpand(nn.Module):
    """Block-diagonal linear projection operating independently per head group."""

    config_class = LinearHeadwiseExpandConfig

    def __init__(self, config: LinearHeadwiseExpandConfig):
        super().__init__()
        self.config = config
        in_features = config.in_features
        num_heads = config.num_heads
        out_per_head = config._out_features // num_heads
        self.weight = nn.Parameter(
            torch.empty(num_heads, out_per_head, in_features // num_heads),
            requires_grad=config.trainable_weight,
        )
        if config.bias:
            self.bias = nn.Parameter(
                torch.empty(config._out_features),
                requires_grad=config.trainable_bias,
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        nn.init.normal_(self.weight.data, mean=0.0, std=sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], self.config.num_heads, -1)
        x = torch.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.config.in_features}, "
            f"num_heads={self.config.num_heads}, "
            f"expand_factor_up={self.config.expand_factor_up}, "
            f"bias={self.config.bias}"
        )
