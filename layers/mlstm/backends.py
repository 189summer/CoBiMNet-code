# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import math

import torch


def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: torch.Tensor = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """Parallel stabilized mLSTM pass over (B, NH, S, DH) tensors."""
    B, NH, S, DH = queries.shape
    dtype, device = queries.dtype, queries.device

    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)
    if (
        lower_triangular_matrix is None
        or lower_triangular_matrix.size(-1) != S
        or lower_triangular_matrix.device != device
    ):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=dtype, device=device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)
    log_fg_diff = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)
    log_fg_matrix = torch.where(ltr, log_fg_diff[:, :, 1:, 1:], -float("inf"))

    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)

    D_matrix = torch.exp(log_D_matrix - max_log_D)
    keys_scaled = keys / math.sqrt(DH)
    C_matrix = (queries @ keys_scaled.transpose(-2, -1)) * D_matrix
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))
    C_matrix_normalized = C_matrix / (normalizer + eps)
    return C_matrix_normalized @ values


def bidirectional_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: torch.Tensor = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """Shared-projection forward and reverse mLSTM passes averaged along the variable axis."""
    opts = dict(
        lower_triangular_matrix=lower_triangular_matrix,
        stabilize_rowwise=stabilize_rowwise,
        eps=eps,
        **kwargs,
    )
    forward = parallel_stabilized_simple(queries, keys, values, igate_preact, fgate_preact, **opts)
    backward = parallel_stabilized_simple(
        queries.flip(-2),
        keys.flip(-2),
        values.flip(-2),
        igate_preact.flip(-2),
        fgate_preact.flip(-2),
        **opts,
    ).flip(-2)
    return 0.5 * (forward + backward)
