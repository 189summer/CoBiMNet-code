# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

import torch
from torch import nn

from .cell import mLSTMCell, mLSTMCellConfig
from .components.conv import CausalConv1d, CausalConv1dConfig
from .components.init import small_init_init_, wang_init_
from .components.linear_headwise import LinearHeadwiseExpand, LinearHeadwiseExpandConfig
from .utils import UpProjConfigMixin


@dataclass
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 8
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 1.0
    embedding_dim: int = -1
    bias: bool = True
    dropout: float = 0.4
    context_length: int = 512
    _num_blocks: int = 1
    _inner_embedding_dim: int = None

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        self._inner_embedding_dim = self._proj_up_dim


class mLSTMLayer(nn.Module):
    config_class = mLSTMLayerConfig

    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config
        inner_dim = config._inner_embedding_dim

        self.proj_up = nn.Linear(config.embedding_dim, 2 * inner_dim, bias=config.bias)

        num_proj_heads = round(inner_dim // config.qkv_proj_blocksize)
        proj_cfg = LinearHeadwiseExpandConfig(
            in_features=inner_dim,
            num_heads=num_proj_heads,
            bias=config.bias,
        )
        self.q_proj = LinearHeadwiseExpand(proj_cfg)
        self.k_proj = LinearHeadwiseExpand(proj_cfg)
        self.v_proj = LinearHeadwiseExpand(proj_cfg)

        self.conv1d = CausalConv1d(
            CausalConv1dConfig(feature_dim=inner_dim, kernel_size=config.conv1d_kernel_size)
        )
        self.conv_act_fn = nn.SiLU()
        self.mlstm_cell = mLSTMCell(
            mLSTMCellConfig(
                context_length=config.context_length,
                embedding_dim=inner_dim,
                num_heads=config.num_heads,
            )
        )
        self.ogate_act_fn = nn.SiLU()
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, config.embedding_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.split(x_inner, self.config._inner_embedding_dim, dim=-1)

        x_conv = self.conv_act_fn(self.conv1d(x_mlstm))
        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_mlstm)

        h_tilde = self.mlstm_cell(q=q, k=k, v=v)
        h_skip = h_tilde + self.learnable_skip * x_conv
        h_state = h_skip * self.ogate_act_fn(z)
        return self.dropout(self.proj_down(h_state))

    def reset_parameters(self):
        small_init_init_(self.proj_up.weight, dim=self.config.embedding_dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim=self.config.embedding_dim, num_blocks=self.config._num_blocks)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        for proj in (self.q_proj, self.k_proj, self.v_proj):
            small_init_init_(proj.weight, dim=self.config.embedding_dim)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        self.mlstm_cell.reset_parameters()
