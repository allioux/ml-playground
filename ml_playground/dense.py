import math
from typing import Optional, Dict, Any

import torch
from torch import Size, Tensor, device, dtype, nn
from torch.nn import init

from ml_playground.utils import activations

class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "identity",
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(Size((out_features, in_features)), **kwargs)
        ).to(device)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kwargs)).to(device)
        else:
            self.bias = None
        if activation in activations.keys():
            self.activation = activations[activation]
        else:
            self.activation = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x_orig_shape = x.shape
        assert x_orig_shape[-1] == self.in_features

        if x.dim() > 2:
            x = x.reshape((-1, self.in_features))

        y = torch.matmul(x, self.weight.mT)

        if self.bias is not None:
            y = y + self.bias

        if self.activation is not None:
            y = self.activation(y)

        if len(x_orig_shape) > 2:
            y = y.reshape(x_orig_shape[:-1] + (self.out_features,))

        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f"bias={self.bias is not None}, activation={self.activation}"
        )
