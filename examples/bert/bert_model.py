import torch
from torch import nn, device, dtype, Tensor, BoolTensor
from torch.nn.modules.module import T

from ml_playground.transformer import Encoder, Dropout, Dense
from ml_playground.utils import check_shape
from typing import Optional, Dict, Any
from examples.model import ModelError, Model
import math
import einops


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
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim, **kwargs))
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

        x = self.token_embed(x)
        x += self.segment_embed(segments)
        x += self.pos_encoding

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
        self.dense = Dense(hidden_dim, hidden_dim, activation="relu")
        self.layer_nom = nn.LayerNorm(hidden_dim, **kwargs)
        self.linear = Dense(hidden_dim, num_embeddings, activation="relu")

    def forward(self, x):
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

    def forward(self, x):
        return self.linear(x[:, 0, :])


class BERTModel(Model[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]):
    def __init__(
        self,
        tokenizer: Any,
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
        self.tokenizer = tokenizer
        self.encoder = BERTEncoder(
            num_embeddings,
            embed_dim,
            ff_hidden_dim,
            num_layers,
            num_heads,
            max_len,
            segments,
            dropout,
            **kwargs
        )
        self.mlm = MaskLM(embed_dim, num_embeddings, **kwargs)
        self.nsp = NextSentencePred(embed_dim, **kwargs)
        # self.linear = Dense(embed_dim, num_embeddings, **kwargs)
        # weight sharing
        # self.linear.weight = self.embed_tgt.weight

    def forward(self, x, segments):
        mask = (
            einops.repeat(x, "batch seq -> batch seq seq", seq=x.dim(-1))
            == self.tokenizer.pad_token_id
        )

        x, attn_weights = self.encoder(x, segments, mask)
        mlm_pred = self.mlm(x)
        nsp_pred = self.nsp(x)
        return mlm_pred, nsp_pred, attn_weights

    def training_step(
        self, x: tuple[Tensor, Tensor], y: Tensor, with_accuracy: bool
    ) -> ModelError:
        """x is a tuple (inputs, segments)
        inputs shape: (batch, seq)
        segments shape: (batch, seq)
        y is a tuple (mlm_tgt, nsp_tgt)
        mlm_tgt shape: (batch, 2)
        nsp_tgt shape: (batch, seq)"""
        inputs, segments = x
        mlm_tgt, nsp_tgt = y
        inputs, segments, y = (
            inputs.to(self.device, non_blocking=True),
            segments.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )

        vocab_size = inputs.size(-1)

        mlm_pred, nsp_pred, _ = self(x, segments)

        mlm_pred = mlm_pred.reshape((-1, vocab_size))

        y = y.reshape((-1,))

        mlm_loss = nn.functional.cross_entropy(
            mlm_pred,
            mlm_tgt,
            ignore_index=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else -100
            ),
        )

        nsp_loss = nn.functional.cross_entropy(
            nsp_pred,
            nsp_tgt,
            ignore_index=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else -100
            ),
        )

        loss = mlm_loss + nsp_loss

        """if with_accuracy:
            pad_mask = y == self.tokenizer.pad_token_id

            size = len(y) - pad_mask.type(torch.int).sum().item()

            accuracy = (pred.argmax(-1) == y).type(torch.float).masked_fill(
                pad_mask, 0.0
            ).sum().item() / size
        else:"""
        accuracy = None

        return {"loss": loss, "accuracy": accuracy}

    def test_step(self, x: T, y: Tensor) -> ModelError:
        return NotImplemented
