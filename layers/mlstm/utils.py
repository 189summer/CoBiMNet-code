# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import math
from dataclasses import dataclass


@dataclass
class UpProjConfigMixin:
    proj_factor: float = None
    round_proj_up_dim_up: bool = True
    round_proj_up_to_multiple_of: int = 64
    _proj_up_dim: int = None

    def _set_proj_up_dim(self, embedding_dim: int) -> None:
        if self.proj_factor is not None and embedding_dim is not None:
            proj_up_dim = self.proj_factor * embedding_dim
            multiplier = proj_up_dim / self.round_proj_up_to_multiple_of
            multiplier = math.ceil(multiplier) if self.round_proj_up_dim_up else math.floor(multiplier)
            self._proj_up_dim = int(multiplier * self.round_proj_up_to_multiple_of)
