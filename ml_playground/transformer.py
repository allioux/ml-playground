import torch
from torch import nn, Tensor, BoolTensor, device, dtype, Size
from ml_playground.dense import Dense
from ml_playground.dropout import Dropout
from ml_playground.utils import masked_softmax, check_shape, PrioritizedItem
import math
from typing import Optional, Dict, Any, Callable


class DotProductAttention(nn.Module):
    def __init__(
        self,
        dropout: float,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.dropout = Dropout(dropout, **kwargs)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[BoolTensor] = None,
    ) -> tuple[Tensor, Tensor]:
        check_shape(query, "... q_seq d")
        check_shape(key, "... kv_seq d", d=query.size(-1))
        check_shape(value, "... kv_seq v", kv_seq=key.size(-2))
        (
            check_shape(
                mask, "... q_seq kv_seq", q_seq=query.size(-2), kv_seq=key.size(-2)
            )
            if mask is not None
            else None
        )

        is_batched = query.dim() == 3

        d = query.size(-1)
        # qk shape: (query, kv)
        if is_batched:
            qk = torch.bmm(query, key.transpose(1, 2))
        else:
            qk = torch.matmul(query, key.t())

        scaled_logits = qk / math.sqrt(d)

        # attention_weights shape: (query, kv)
        attention_weights = masked_softmax(scaled_logits, mask, dim=-1)

        # output shape: (query, value)
        if is_batched:
            out = torch.bmm(self.dropout(attention_weights), value)
        else:
            out = torch.matmul(self.dropout(attention_weights), value)

        return out, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim if key_dim is not None else query_dim
        self.value_dim = value_dim if value_dim is not None else query_dim

        self.queries_layers = Dense(query_dim, self.value_dim, **kwargs)
        self.keys_layers = Dense(self.key_dim, self.value_dim, **kwargs)
        self.values_layers = Dense(self.value_dim, self.value_dim, **kwargs)

        self.attention_layer = DotProductAttention(dropout=dropout, **kwargs)

        self.linear = Dense(self.value_dim, self.value_dim, **kwargs)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[BoolTensor] = None,
        average_weights: Optional[bool] = False,
    ) -> tuple[Tensor, Tensor]:
        check_shape(query, "... q_seq q_dim", q_dim=self.query_dim)
        check_shape(key, "... kv_seq k_dim", k_dim=self.key_dim)
        check_shape(
            value, "... kv_seq v_dim", kv_seq=key.size(-2), v_dim=self.value_dim
        )
        (
            check_shape(
                mask, "... q_seq kv_seq", q_seq=query.size(-2), kv_seq=key.size(-2)
            )
            if mask is not None
            else None
        )

        is_batched = query.dim() == 3

        q = self.transpose_input(self.queries_layers(query))
        k = self.transpose_input(self.keys_layers(key))
        v = self.transpose_input(self.values_layers(value))

        # att shape: (batch_dim, queries_num, values_dim)
        att, weights = self.attention_layer(q, k, v, mask)

        if is_batched:
            weights = weights.reshape((-1, self.num_heads) + weights.shape[1:])

        if average_weights:
            weights = weights.mean(dim=-3, keepdim=True)

        att = self.transpose_output(att)
        out = self.linear(att)

        return out, weights

    def transpose_input(self, x: Tensor) -> Tensor:
        x = x.reshape(*x.shape[:-1], self.num_heads, -1)
        x = x.permute(*tuple(range(x.dim()))[:-3], -2, -3, -1)
        return x.reshape(-1, *x.shape[-2:])

    def transpose_output(self, x: Tensor) -> Tensor:
        if x.shape[0] != self.num_heads:
            x = x.reshape(-1, self.num_heads, *x.shape[1:])

        x = x.permute(*tuple(range(x.dim()))[:-3], -2, -3, -1)
        return x.reshape(*x.shape[:-2], -1)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.dropout = Dropout(rate=dropout, **kwargs)
        self.d_model = d_model

        position = torch.arange(max_len, **kwargs).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, **kwargs) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, **kwargs)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        check_shape(x, "... seq d", d=self.d_model)

        seq_len = x.shape[-2]
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.relu = Dense(embed_dim, hidden_dim, activation="relu", **kwargs)
        self.linear = Dense(hidden_dim, embed_dim, **kwargs)
        self.dropout = Dropout(dropout, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(x)
        x = self.linear(x)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.mha1 = MultiHeadAttention(embed_dim, num_heads, dropout, **kwargs)
        self.mha2 = MultiHeadAttention(embed_dim, num_heads, dropout, **kwargs)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout, **kwargs)

        self.lnorm1 = nn.LayerNorm(embed_dim, **kwargs)
        self.lnorm2 = nn.LayerNorm(embed_dim, **kwargs)
        self.lnorm3 = nn.LayerNorm(embed_dim, **kwargs)

        self.dropout = Dropout(dropout, **kwargs)

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        mask: Optional[BoolTensor] = None,
        state_mask: Optional[BoolTensor] = None,
        average_weights: Optional[bool] = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        check_shape(x, "... seq1 emb")
        check_shape(state, "... seq2 emb")
        (
            check_shape(mask, "... seq11 seq12", seq11=x.size(-2), seq12=x.size(-2))
            if mask is not None
            else None
        )
        (
            check_shape(
                state_mask, "... seq21 seq22", seq21=x.size(-2), seq22=state.size(-2)
            )
            if state_mask is not None
            else None
        )

        y, self_attn_weights = self.mha1(x, x, x, mask, average_weights)
        y = self.lnorm1(x + self.dropout(y))
        z, cross_attn_weights = self.mha2(y, state, state, state_mask, average_weights)
        v = self.lnorm2(y + self.dropout(z))
        out = self.lnorm3(z + self.ff(v))

        return out, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.dec_layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, ff_hidden_dim, num_heads, dropout, **kwargs)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        mask: Optional[BoolTensor] = None,
        state_mask: Optional[BoolTensor] = None,
        average_weights: Optional[bool] = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        check_shape(x, "... seq1 emb")
        check_shape(state, "... seq2 emb")
        (
            check_shape(mask, "... seq11 seq12", seq11=x.size(-2), seq12=x.size(-2))
            if mask is not None
            else None
        )
        (
            check_shape(
                state_mask, "... seq21 seq22", seq21=x.size(-2), seq22=state.size(-2)
            )
            if state_mask is not None
            else None
        )

        x_length = x.shape[-2]
        state_length = state.shape[-2]
        is_batched = x.dim() == 3
        batch_dim = Size((x.shape[0],) if is_batched else ())

        self_attn_weights = torch.empty(
            (self.num_layers,) + batch_dim + (self.num_heads, x_length, x_length),
            device=self.device,
        )

        cross_attn_weights = torch.empty(
            (self.num_layers,) + batch_dim + (self.num_heads, x_length, state_length),
            device=self.device,
        )

        for i, dec_layer in enumerate(self.dec_layers):
            x, self_attn_weights[i], cross_attn_weights[i] = dec_layer(
                x, state, mask, state_mask, average_weights
            )

        if is_batched:
            self_attn_weights.transpose_(0, 1)
            cross_attn_weights.transpose_(0, 1)

        attn_weights = {
            "self_attn_weights": self_attn_weights,
            "cross_attn_weights": cross_attn_weights,
        }

        return x, attn_weights


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout, **kwargs)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout, **kwargs)

        self.lnorm1 = nn.LayerNorm(embed_dim, **kwargs)
        self.lnorm2 = nn.LayerNorm(embed_dim, **kwargs)

        self.dropout = Dropout(dropout, **kwargs)

    def forward(
        self,
        x: Tensor,
        mask: Optional[BoolTensor] = None,
        average_weights: Optional[bool] = False,
    ) -> tuple[Tensor, Tensor]:
        check_shape(x, "... seq emb", emb=self.embed_dim)
        (
            check_shape(
                mask,
                "... seq1 seq2",
                seq1=x.size(-2),
                seq2=x.size(-2),
            )
            if mask is not None
            else ()
        )

        y, attn_weights = self.mha(x, x, x, mask, average_weights)
        y = self.lnorm1(x + self.dropout(y))
        z = self.lnorm2(y + self.ff(y))

        return z, attn_weights


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim, ff_hidden_dim, num_heads, dropout=dropout, **kwargs
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[BoolTensor] = None,
        average_weights: Optional[bool] = False,
    ) -> tuple[Tensor, Tensor]:
        check_shape(x, "... seq emb", emb=self.embed_dim)
        (
            check_shape(
                mask,
                "... seq1 seq2",
                seq1=x.size(-2),
                seq2=x.size(-2),
            )
            if mask is not None
            else ()
        )

        seq_length = x.shape[-2]
        is_batched = x.dim() == 3

        batch_dim = Size((x.shape[0],) if is_batched else ())

        attn_weights = torch.empty(
            (self.num_layers,) + batch_dim + (self.num_heads, seq_length, seq_length),
            device=self.device,
        )

        for i, enc_layer in enumerate(self.enc_layers):
            x, attn_weights[i] = enc_layer(x, mask, average_weights)

        if is_batched:
            attn_weights.transpose_(0, 1)

        return x, attn_weights


class Transformer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        ff_hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.embed_src = nn.Embedding(num_embeddings, embed_dim, **kwargs)
        self.embed_tgt = nn.Embedding(num_embeddings, embed_dim, **kwargs)
        self.encoder = Encoder(
            embed_dim, ff_hidden_dim, num_layers, num_heads, dropout, **kwargs
        )
        self.decoder = Decoder(
            embed_dim, ff_hidden_dim, num_layers, num_heads, dropout, **kwargs
        )
        self.linear = Dense(embed_dim, num_embeddings, **kwargs)
        # weight sharing
        self.linear.weight = self.embed_tgt.weight
        self.pos_encoding_src = PositionalEncoding(embed_dim, dropout, **kwargs)
        self.pos_encoding_tgt = PositionalEncoding(embed_dim, dropout, **kwargs)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[BoolTensor] = None,
        tgt_mask: Optional[BoolTensor] = None,
        state_mask: Optional[BoolTensor] = None,
        average_weights: Optional[bool] = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        check_shape(src, "... seq1")
        check_shape(tgt, "... seq2")
        (
            check_shape(
                src_mask, "... seq11 seq12", seq11=src.size(-1), seq12=src.size(-1)
            )
            if src_mask is not None
            else None
        )
        (
            check_shape(
                tgt_mask, "... seq21 seq22", seq21=tgt.size(-1), seq22=tgt.size(-1)
            )
            if tgt_mask is not None
            else None
        )
        (
            check_shape(
                state_mask, "... seq2 seq1", seq1=src.size(-1), seq2=tgt.size(-1)
            )
            if state_mask is not None
            else None
        )

        src = self.embed_src(src)
        src = self.pos_encoding_src(src)
        state, enc_attn_weights = self.encoder(src, src_mask, average_weights)

        tgt = self.embed_tgt(tgt)
        tgt = self.pos_encoding_tgt(tgt)
        tgt, dec_attn_weights = self.decoder(
            tgt, state, tgt_mask, state_mask, average_weights
        )

        out = self.linear(tgt)

        attn_weights = {"enc_attn_weights": enc_attn_weights} | {
            "dec_self_attn_weights": dec_attn_weights["self_attn_weights"],
            "dec_cross_attn_weights": dec_attn_weights["cross_attn_weights"],
        }

        return out, attn_weights


def beam_search(
    f: Callable[[Tensor], Tensor],
    init_state: Tensor,
    start_idx: int,
    stop_condition: Callable[[int], bool],
    beam_size: int = 1,
    alpha: float = 0.75,
) -> Tensor:
    """Beam search for transformer-based models"""

    total_length = len(init_state)

    states = [PrioritizedItem(0, (start_idx, init_state))]

    for _ in range(start_idx, total_length - 1):
        acc: list[PrioritizedItem] = []
        for state_score, (last_idx, state) in states:
            if stop_condition(state[last_idx]):
                acc.append(PrioritizedItem(state_score, (last_idx, state)))
                continue

            top_preds = f(state)[last_idx].topk(beam_size)
            top_preds = [
                (prob.item(), idx)
                for prob, idx in zip(list(top_preds[0]), list(top_preds[1]))
                if prob > 0.0
            ]
            next_idx = last_idx + 1

            for prob, idx in top_preds:

                n = next_idx - start_idx + 1
                new_state_score = (
                    state_score * ((n - 1) ** alpha) + math.log(prob)
                ) / (n**alpha)
                new_state = state.detach().clone()
                new_state[next_idx] = idx
                acc.append(PrioritizedItem(new_state_score, (next_idx, new_state)))

        states = sorted(acc, key=lambda pred: pred.priority, reverse=True)[:beam_size]

    _, (_, final_state) = max(states)

    return final_state
