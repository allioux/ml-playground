from typing import Optional, Any

import torch
from torch import nn, dtype, device, Tensor
import torch.nn.functional as F

from ml_playground.dense import Dense
from ml_playground.transformer import (
    Encoder,
    PositionalEncoding,
    beam_search as beam_search_transformer,
)
from ml_playground.utils import TorchKw
from examples.model import ModelError, Model


class TransformerLM(Model[Tensor, Tensor]):
    def __init__(
        self,
        tokenizer: Any,
        embedding_dim: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: TorchKw = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.ff_hidden_dim = ff_hidden_dim
        self.tokenizer = tokenizer
        num_embeddings = len(tokenizer)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
        self.num_heads = 8
        self.num_layers = 6
        self.encoder = Encoder(
            embedding_dim,
            ff_hidden_dim,
            "relu",
            self.num_layers,
            self.num_heads,
            dropout,
            **kwargs
        )
        self.linear = Dense(embedding_dim, num_embeddings, **kwargs)
        self.linear.weight = self.embedding.weight
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout, **kwargs)

    def forward(
        self,
        x: Tensor,
        average_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:

        mask = torch.ones((x.size(-1), x.size(-1)), device=self.device).tril() == 0.0

        x = self.embedding(x)
        x = self.pos_encoding(x)
        x, attn_weights = self.encoder(x, mask, average_weights)
        x = self.linear(x)

        return x, attn_weights

    def forward_norm(
        self,
        x: Tensor,
        average_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:

        x, attn_weights = self.forward(x, average_weights)

        return F.softmax(x, dim=-1), attn_weights

    def predict(
        self, prefix: str, max_length: int, beam_size: int = 1, alpha: float = 0.75
    ) -> Optional[str]:
        if self.tokenizer.bos_token_id is None or self.tokenizer.pad_token_id is None:
            return None

        self.eval()

        prefix_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prefix)

        init_state = torch.full(
            (len(prefix_ids) + max_length,),
            self.tokenizer.pad_token_id,
            device=self.device,
        )
        init_state[: len(prefix_ids)] = torch.tensor(prefix_ids, device=self.device)

        final_state = beam_search_transformer(
            lambda x: self.forward_norm(x)[0],
            init_state,
            len(prefix_ids) - 1,
            lambda id: id == self.tokenizer.eos_token_id,
            beam_size,
            alpha,
        )

        eos_idx = [
            idx for idx, v in enumerate(final_state) if v == self.tokenizer.eos_token_id
        ]
        end_idx = eos_idx[0] if len(eos_idx) > 0 else len(final_state)

        return self.tokenizer.decode(final_state[1:end_idx])

    def common_step(
        self, x: Tensor, y: Tensor, with_accuracy: bool = False
    ) -> ModelError:
        x, y = x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )
        pad_token_id = self.tokenizer.pad_token_id
        pred, _ = self(x)

        vocab_size = pred.size(-1)
        pred = pred.reshape((-1, vocab_size))
        y = y.reshape((-1,))

        loss = nn.functional.cross_entropy(
            pred, y, ignore_index=pad_token_id if pad_token_id is not None else -100
        )

        if with_accuracy:
            pad_mask = y == self.tokenizer.pad_token_id

            size = len(y) - pad_mask.type(torch.int).sum().item()

            accuracy = (pred.argmax(-1) == y).type(torch.float).masked_fill(
                pad_mask, 0.0
            ).sum().item() / size
        else:
            accuracy = None

        return {"loss": loss, "accuracy": accuracy}

    def training_step(
        self, x: Tensor, y: Tensor, with_accuracy: bool = False
    ) -> ModelError:
        self.train()
        return self.common_step(x, y, with_accuracy)

    def test_step(self, x: Tensor, y: Tensor) -> ModelError:
        self.eval()
        return self.common_step(x, y, with_accuracy=True)
