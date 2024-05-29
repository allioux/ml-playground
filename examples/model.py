from typing import Optional, Protocol, TypedDict, TypeVar 

from torch import Tensor, nn

T = TypeVar("T", contravariant=True)


class ModelError(TypedDict):
    loss: Tensor
    accuracy: Optional[float]


class ModelAux(Protocol[T]):
    def training_step(self, x: T, y: Tensor, with_accuracy: bool) -> ModelError: ...
    def test_step(self, x: T, y: Tensor) -> ModelError: ...


class Model(nn.Module, ModelAux[T]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
