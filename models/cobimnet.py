from typing import Sequence

import torch
from torch import nn

from layers.cobim import CoBiM
from layers.cyphe import CyPHE
from layers.revin import RevIN


class CoBiMNet(nn.Module):
    """Cycle-Enhanced Bidirectional Memory Network for multivariate time series forecasting."""

    def __init__(
        self,
        seq_len: int = 720,
        pred_len: int = 96,
        enc_in: int = 7,
        d_model: int = 512,
        cycle_len: int = 24,
        period_len: int = 24,
        gate_init: float = 0.3,
        windows: Sequence[int] = (24, 48, 72, 144),
        num_heads: int = 4,
        conv1d_kernel_size: int = 8,
        qkv_proj_blocksize: int = 4,
        dropout: float = 0.4,
        use_cycle: bool = True,
        use_cobim: bool = True,
        bidirectional: bool = True,
        use_context: bool = True,
        a24_readout: str = "attention",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        self.revin = RevIN(enc_in)
        self.cyphe = CyPHE(
            seq_len=seq_len,
            enc_in=enc_in,
            cycle_len=cycle_len,
            period_len=period_len,
            gate_init=gate_init,
            windows=windows,
            d_model=d_model,
            use_cycle=use_cycle,
            a24_readout=a24_readout,
        )
        self.cobim = (
            CoBiM(
                d_model=d_model,
                context_length=max(512, enc_in),
                num_heads=num_heads,
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=qkv_proj_blocksize,
                proj_factor=1.0,
                dropout=dropout,
                bidirectional=bidirectional,
                use_context=use_context,
            )
            if use_cobim
            else nn.Identity()
        )
        self.head = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        x_norm, reverse_revin = self.revin(x)
        h = self.cyphe(x_norm, cycle_index)
        z = self.cobim(h)
        y = self.head(z).permute(0, 2, 1)
        return reverse_revin(y)

    def parameter_groups(
        self,
        backbone_lr: float,
        cycle_lr: float,
        cycle_weight_decay: float = 1e-4,
    ) -> list[dict]:
        cycle_mod = self.cyphe.cycle_enhancement
        cycle_ids = {id(p) for p in cycle_mod.parameters()} if cycle_mod is not None else set()

        backbone_params = [p for p in self.parameters() if p.requires_grad and id(p) not in cycle_ids]
        cycle_params = [p for p in self.parameters() if p.requires_grad and id(p) in cycle_ids]

        groups = [{"params": backbone_params, "lr": backbone_lr, "weight_decay": 0.0}]
        if cycle_params:
            groups.append({"params": cycle_params, "lr": cycle_lr, "weight_decay": cycle_weight_decay})
        return groups
