from dataclasses import dataclass, field, astuple
from typing import Any, Callable, Optional, TypedDict

from einops import rearrange
from torch import BoolTensor, Tensor, device, dtype
import torch.nn.functional as F


activations: dict[str, Callable[..., Any]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
}


def check_shape(tensor: Tensor, pattern: str, **kwargs) -> Tensor:
    """Check the shape of a tensor by applying the identity transformation with a certain
    pattern. Hopefully it does not hurt the performances too much…"""
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


def masked_softmax(x: Tensor, mask: Optional[BoolTensor] = None, dim: int = -1):
    if mask is None:
        logits = x
    else:
        logits = x.masked_fill(mask, float("-inf"))

    return F.softmax(logits, dim)


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)

    def __iter__(self):
        return iter(astuple(self))


class TorchKw(TypedDict):
    dtype: Optional[dtype]
    device: Optional[device]
