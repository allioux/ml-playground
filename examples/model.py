from typing import Optional, Protocol, TypedDict, TypeVar

from torch import Tensor, nn

In = TypeVar("In", contravariant=True)
Out = TypeVar("Out", contravariant=True)


class ModelError(TypedDict):
    loss: Tensor
    accuracy: Optional[float]


class ModelAux(Protocol[In, Out]):
    def training_step(self, x: In, y: Out, with_accuracy: bool) -> ModelError: ...
    def test_step(self, x: In, y: Out) -> ModelError: ...


class Model(nn.Module, ModelAux[In, Out]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
