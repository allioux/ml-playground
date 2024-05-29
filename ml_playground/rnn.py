import math
from typing import Any, Callable, Optional, TypeVar

import torch
from torch import nn, Tensor, device, dtype, Size
import torch.nn.functional as F

from ml_playground.dense import Dense
from ml_playground.dropout import Dropout
from ml_playground.utils import PrioritizedItem, TorchKw


class RNNCell(nn.Module):
    def __init__(
        self,
        inputs_dim: int,
        hidden_dim: int,
        activation: str = "identity",
        bias: bool = True,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ):
        kwargs: TorchKw = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.inputs_dim = inputs_dim
        self.hidden_dim = hidden_dim
        self.dense = Dense(
            inputs_dim + hidden_dim, hidden_dim, bias, activation, **kwargs
        )

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tensor:
        is_batched = x.dim() == 2
        batch_dim = Size((x.size(0),) if is_batched else ())

        if h0 is None:
            h0 = torch.zeros(batch_dim + (self.hidden_dim,), device=self.device)

        xh = torch.cat((x, h0), dim=-1)
        return self.dense(xh)

    def get_output(self, state: Tensor) -> Tensor:
        return state


class LSTMCell(nn.Module):
    def __init__(
        self,
        inputs_dim: int,
        hidden_dim: int,
        bias: bool = True,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ):
        kwargs: TorchKw = {"device": device, "dtype": dtype}
        super().__init__()

        self.inputs_dim = inputs_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        self.forget = Dense(
            inputs_dim + hidden_dim, hidden_dim, bias, "sigmoid", **kwargs
        )
        self.input = Dense(
            inputs_dim + hidden_dim, hidden_dim, bias, "sigmoid", **kwargs
        )
        self.candidates = Dense(
            inputs_dim + hidden_dim, hidden_dim, bias, "tanh", **kwargs
        )
        self.output = Dense(
            inputs_dim + hidden_dim, hidden_dim, bias, "sigmoid", **kwargs
        )

    def forward(self, x: Tensor, hc0: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        is_batched = x.dim() == 2
        batch_dim = Size((x.size(0),) if is_batched else ())

        if hc0 is None:
            h0 = torch.zeros(batch_dim + (self.hidden_dim,), device=self.device)
            c0 = torch.zeros(batch_dim + (self.hidden_dim,), device=self.device)
        else:
            h0, c0 = hc0

        xh = torch.cat((x, h0), dim=-1)
        f = self.forget(xh)
        i = self.input(xh)
        ct = self.candidates(xh)

        c = f * c0 + i * ct
        h = self.output(xh) * F.tanh(c)

        return h, c

    def get_output(self, state: tuple[Tensor, Tensor]) -> Tensor:
        return state[0]


class RNN(nn.Module):
    def __init__(
        self,
        rnn_cell: Any,  # can't be more precise
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ):
        kwargs: TorchKw = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.rnn_cell = rnn_cell

        self.dropout = Dropout(dropout, **kwargs)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> Tensor:
        hidden_dim = self.rnn_cell.hidden_dim
        seq_length = inputs.size(1)
        is_batched = inputs.dim() == 3

        batch_dim = Size((inputs.size(0),) if is_batched else ())
        outputs = torch.empty(
            (seq_length,) + batch_dim + (hidden_dim,), device=self.device
        )

        if is_batched:
            inputs.transpose_(0, 1)

        for i in range(seq_length):
            state = self.rnn_cell(inputs[i], state)
            out = self.rnn_cell.get_output(state)
            outputs[i] = self.dropout(out)

        if is_batched:
            outputs.transpose_(0, 1)

        return outputs


class BiRNN(nn.Module):
    def __init__(self, rnn_fwd: RNN, rnn_bwd: RNN) -> None:
        super().__init__()
        self.rnn_fwd = rnn_fwd
        self.rnn_bwd = rnn_bwd

    def forward(
        self,
        x: Tensor,
        state_fwd: Optional[Tensor] = None,
        state_bwd: Optional[Tensor] = None,
    ):
        seq_length = x.shape[0]
        hidden_dim = self.rnn_fwd.hidden_dim

        y_fwd = self.rnn_fwd(x, state_fwd)
        y_bwd = self.rnn_bwd(x, state_bwd)

        if len(x.shape) == 2:
            outputs = torch.empty((seq_length, 2 * hidden_dim), device=self.device)
        elif len(x.shape) == 3:
            batch_size = x.shape[1]
            outputs = torch.empty(
                (seq_length, batch_size, 2 * hidden_dim), device=self.device
            )
        else:
            assert False

        outputs[..., :hidden_dim] = y_fwd
        outputs[..., hidden_dim:] = y_bwd

        return outputs


S = TypeVar("S", contravariant=True)


def beam_search(
    f: Callable[[Tensor, S], S],
    out_proj: Callable[[S], Tensor],
    prefix: Tensor,
    max_length: int,
    stop_condition: Callable[[int], bool],
    beam_size: int = 1,
    alpha: float = 0.75,
) -> Tensor:
    """Beam search for RNN-based models. Should be unified with the
    transformer-based one in the future"""

    total_length = len(prefix) + max_length

    seq = torch.empty((total_length,), device=prefix.device)
    seq[: len(prefix)] = torch.tensor(prefix, device=prefix.device)

    state: S = None
    for x in prefix:
        state = f(x, state)

    candidates = [PrioritizedItem(0, (seq, state, len(prefix)))]

    for i in range(len(prefix), total_length - 1):
        acc: list[PrioritizedItem] = []
        for candidate_score, (seq, state, last_idx) in candidates:
            if stop_condition(seq[last_idx]):
                acc.append(PrioritizedItem(candidate_score, (seq, state, last_idx)))
                continue

            new_state = f(seq[last_idx], state)
            out = out_proj(new_state)

            top_preds = out.topk(beam_size, dim=-1)
            top_preds = [
                (prob.item(), idx)
                for prob, idx in zip(list(top_preds[0]), list(top_preds[1]))
                if prob > 0.0
            ]
            next_idx = last_idx + 1

            for prob, idx in top_preds:

                new_candidate_score = (
                    candidate_score * ((next_idx - 1) ** alpha) + math.log(prob)
                ) / (next_idx**alpha)
                new_seq = seq.detach().clone()
                new_seq[next_idx] = idx
                acc.append(
                    PrioritizedItem(new_candidate_score, (new_seq, new_state, next_idx))
                )

        states = sorted(acc, key=lambda pred: pred.priority, reverse=True)[:beam_size]

    _, (final_seq, _, final_idx) = max(states)

    return final_seq[: final_idx + 1]
