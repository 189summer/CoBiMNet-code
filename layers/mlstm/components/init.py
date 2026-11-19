# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import math

import torch


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    assert param.dim() == 1, f"param must be 1-dimensional, got {param.dim()}"
    init_vals = torch.linspace(start, end, param.shape[0])
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int) -> torch.Tensor:
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param
