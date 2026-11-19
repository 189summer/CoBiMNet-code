# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

import torch
from torch import nn

from .backends import bidirectional_stabilized_simple
from .components.init import bias_linspace_init_
from .components.ln import MultiHeadLayerNorm


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1


class mLSTMCell(nn.Module):
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config
        self.backend_fn = bidirectional_stabilized_simple

        self.igate = nn.Linear(3 * config.embedding_dim, config.num_heads)
        self.fgate = nn.Linear(3 * config.embedding_dim, config.num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=config.embedding_dim, weight=True, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, _ = q.shape
        if_gate_input = torch.cat([q, k, v], dim=-1)

        q = q.view(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.view(B, S, self.config.num_heads, -1).transpose(1, 2)
        v = v.view(B, S, self.config.num_heads, -1).transpose(1, 2)

        igate_preact = self.igate(if_gate_input).transpose(-1, -2).unsqueeze(-1)
        fgate_preact = self.fgate(if_gate_input).transpose(-1, -2).unsqueeze(-1)

        h_state = self.backend_fn(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=self.causal_mask,
        )
        h_state_norm = self.outnorm(h_state)
        return h_state_norm.transpose(1, 2).reshape(B, S, -1)

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
