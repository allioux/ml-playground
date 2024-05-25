import torch
from torch import device, nn, Tensor
from typing import Any, Optional
from ml_playground.transformer import Transformer, beam_search
import torch.nn.functional as F
from examples.model import Model, ModelError


class Translator(Model[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        tokenizer: Any,
        dropout: float = 0.1,
        device: Optional[device] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = 4
        self.num_heads = 8
        self.transformer = Transformer(
            len(self.tokenizer),
            128,
            512,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=dropout,
            device=self.device,
        )

    def forward(
        self, src: Tensor, tgt: Tensor, average_weights: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        src_mask, tgt_mask, state_mask = self.create_masks(src, tgt)
        return self.transformer(
            src, tgt, src_mask, tgt_mask, state_mask, average_weights
        )

    def forward_norm(
        self, src: Tensor, tgt: Tensor, average_weights: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        out, attn_weights = self.forward(src, tgt, average_weights)
        return F.softmax(out, dim=-1), attn_weights

    def create_masks(self, src: Tensor, tgt: Tensor):
        is_batched = src.dim() == 2
        src_length = src.size(-1)
        tgt_length = tgt.size(-1)

        src_mask = (src == self.tokenizer.pad_token_id).unsqueeze(-2)
        src_mask = src_mask.repeat_interleave(src_length, dim=-2)
        if is_batched:
            src_mask = src_mask.repeat_interleave(self.num_heads, dim=0)

        tgt_mask = (
            torch.ones((tgt.size(-1), tgt.size(-1)), device=self.device).tril() == 0.0
        )

        state_mask = (src == self.tokenizer.pad_token_id).unsqueeze(-2)
        state_mask = state_mask.repeat_interleave(tgt_length, dim=-2)
        if is_batched:
            state_mask = state_mask.repeat_interleave(self.num_heads, dim=0)

        return src_mask, tgt_mask, state_mask

    def common_step(
        self, x: tuple[Tensor, Tensor], y: Tensor, with_accuracy: bool = False
    ) -> ModelError:
        src, tgt = x
        src, tgt, y = src.to(self.device), tgt.to(self.device), y.to(self.device)

        pad_token_id = self.tokenizer.pad_token_id

        pred, _ = self(src, tgt)

        vocab_size = pred.shape[-1]
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
        self, x: tuple[Tensor, Tensor], y: Tensor, with_accuracy: bool = False
    ) -> ModelError:
        self.train()
        return self.common_step(x, y, with_accuracy)

    def test_step(self, x: tuple[Tensor, Tensor], y: Tensor) -> ModelError:
        self.eval()
        return self.common_step(x, y, with_accuracy=True)

    def predict(
        self,
        src_str: str,
        max_length: int = 50,
        beam_size: int = 1,
        alpha: float = 0.75,
    ) -> Optional[str]:
        self.eval()
        if self.tokenizer.bos_token_id is None or self.tokenizer.pad_token_id is None:
            return None

        src_token_ids = self.tokenizer.encode(src_str)
        src = torch.tensor(src_token_ids, device=self.device)

        init_state = torch.full(
            (max_length,),
            self.tokenizer.pad_token_id,
            device=self.device,
        )
        init_state[0] = self.tokenizer.bos_token_id

        final_state = beam_search(
            lambda tgt: self.forward_norm(src, tgt)[0],
            init_state,
            0,
            lambda id: id == self.tokenizer.eos_token_id,
            beam_size,
            alpha,
        )

        eos_idx = [
            idx for idx, v in enumerate(final_state) if v == self.tokenizer.eos_token_id
        ]
        end_idx = eos_idx[0] if len(eos_idx) > 0 else len(final_state)

        return self.tokenizer.decode(final_state[1:end_idx])
