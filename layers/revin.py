import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible Instance Normalization along the temporal dimension."""

    def __init__(self, num_variates: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1))
        self.beta = nn.Parameter(torch.zeros(num_variates, 1))

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)  # (B, N, L)
        assert x.shape[1] == self.num_variates, (
            f"Expected {self.num_variates} variates, got {x.shape[1]}"
        )

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var_rsqrt = var.clamp(min=self.eps).rsqrt()
        rescaled = (x - mean) * var_rsqrt * self.gamma + self.beta

        def reverse_fn(scaled_output: torch.Tensor) -> torch.Tensor:
            y = scaled_output.transpose(1, 2)
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min=self.eps)
            unscaled = (y - self.beta) / clamped_gamma
            out = unscaled * var.sqrt() + mean
            return out.transpose(1, 2)

        return rescaled.transpose(1, 2), reverse_fn
