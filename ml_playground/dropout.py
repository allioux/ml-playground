import torch
from torch import nn, device, dtype, Tensor
from typing import Optional


class Dropout(nn.Module):
    def __init__(
        self,
        rate: float,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.device = device
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.rate > 0.0:
            probs = torch.empty(x.shape, device=self.device, dtype=self.dtype).fill_(
                1 - self.rate
            )
            mask = torch.bernoulli(probs)
            return mask * x
        else:
            return x
