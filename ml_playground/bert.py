from typing import Optional, Dict, Any

import torch
from torch import nn, device, dtype, Tensor, BoolTensor

from ml_playground.transformer import Encoder, Dense
from ml_playground.utils import check_shape


class BERTEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        ff_hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        max_len: int = 5000,
        segments: int = 1,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ):
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.token_embed = nn.Embedding(num_embeddings, embed_dim, **kwargs)
        self.segment_embed = nn.Embedding(segments, embed_dim, **kwargs)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, embed_dim, **kwargs))
        self.transformer = Encoder(
            embed_dim, ff_hidden_dim, "gelu", num_layers, num_heads, dropout, **kwargs
        )
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        x: Tensor,
        segments: Tensor,
        mask: Optional[BoolTensor] = None,
        average_weights: Optional[bool] = False,
    ):
        check_shape(x, "... seq")
        check_shape(
            segments,
            "... seq",
            seq=x.size(-1),
        )
        (
            check_shape(
                mask,
                "... seq1 seq2",
                seq1=x.size(-1),
                seq2=x.size(-1),
            )
            if mask is not None
            else ()
        )

        x = self.token_embed(x)
        x += self.segment_embed(segments)
        x += self.pos_encoding[: x.size(-2)]

        out, attn_weights = self.transformer(x, mask, average_weights)

        return out, attn_weights


class MaskLM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_embeddings: int,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ):
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.dense = Dense(hidden_dim, hidden_dim, activation="relu", **kwargs)
        self.layer_nom = nn.LayerNorm(hidden_dim, **kwargs)
        self.linear = Dense(hidden_dim, num_embeddings, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.layer_nom(x)
        return self.linear(x)


class NextSentencePred(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ):
        super().__init__()
        self.linear = Dense(hidden_dim, 2, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x[:, 0])


class BERTModel(nn.Module):
    """BERT model equipped with masked-language modeling and next-sentence
    prediction layers for pretraining."""

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int = 768,
        ff_hidden_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        max_len: int = 5000,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ):
        kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.encoder = BERTEncoder(
            num_embeddings,
            embed_dim,
            ff_hidden_dim,
            num_layers,
            num_heads,
            max_len,
            segments=2,
            dropout=dropout,
            **kwargs
        )
        self.mlm = MaskLM(embed_dim, num_embeddings, **kwargs)
        self.nsp = NextSentencePred(embed_dim, **kwargs)

    def forward(
        self,
        x: Tensor,
        segments: Tensor,
        mlm_mask: BoolTensor,
        pad_mask: Optional[BoolTensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        check_shape(x, "... seq")
        check_shape(
            segments,
            "... seq",
            seq=x.size(-1),
        )
        check_shape(
            mlm_mask,
            "... seq",
            seq=x.size(-1),
        )
        (
            check_shape(
                pad_mask,
                "... seq1 seq2",
                seq1=x.size(-1),
                seq2=x.size(-1),
            )
            if pad_mask is not None
            else ()
        )
        x, attn_weights = self.encoder(x, segments, pad_mask)
        mlm_pred = self.mlm(x[mlm_mask])
        nsp_pred = self.nsp(x)
        return mlm_pred, nsp_pred, attn_weights
