import torch
from torchinfo import summary
from transformers import AutoTokenizer
from examples.language_model.tiny_stories import TinyStoriesDS
from examples.trainer import Trainer
from torch.utils.data import DataLoader
from typing import Optional, Any
from torch import device, Tensor
from pathlib import Path
from examples.language_model.transformer_lm import TransformerLM
import transformers


class Collate:
    def __init__(
        self,
        tokenizer: Any,
        seq_length: Optional[int] = None,
        device: Optional[device] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = "cpu"

        self.seq_length = seq_length

    def collate(self, samples: list[str]) -> tuple[Tensor, Tensor]:
        samples_token_ids: list[list[int]] = []

        pad_token_id = self.tokenizer.pad_token_id
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        for sample in samples:
            token_ids = self.tokenizer.encode(sample)
            samples_token_ids.append(token_ids)

        batch_size = len(samples_token_ids)

        if self.seq_length is None:
            seq_length = len(max(samples_token_ids, key=lambda x: len(x))) + 1
        else:
            seq_length = self.seq_length

        src_ids = torch.empty(
            (batch_size, seq_length), dtype=torch.int64, device=self.device
        )
        tgt_ids = torch.empty(
            (batch_size, seq_length), dtype=torch.int64, device=self.device
        )

        for batch, token_ids in enumerate(samples_token_ids):
            length = min(seq_length - 1, len(token_ids))
            pad_length = seq_length - 1 - length
            src_ids[batch] = torch.tensor(
                [bos_token_id] + token_ids[:length] + pad_length * [pad_token_id],
                dtype=torch.int64,
                device=self.device,
            )
            tgt_ids[batch] = torch.tensor(
                token_ids[:length] + [eos_token_id] + pad_length * [pad_token_id],
                dtype=torch.int64,
                device=self.device,
            )

        return src_ids, tgt_ids


class LMExample:
    def __init__(
        self,
        checkpoint_file: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        transformers.logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "examples/language_model/tokenizer"
        )

        self.language_model = TransformerLM(
            self.tokenizer, 256, 1024, device=self.device, dropout=0.2
        )

        optimizer = torch.optim.Adam(self.language_model.parameters())
        starting_epoch = 0
        starting_sample = 0

        if checkpoint_file is not None and Path(checkpoint_file).is_file():
            checkpoint = torch.load(checkpoint_file)
            self.language_model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            starting_epoch = checkpoint["epoch"]
            starting_sample = checkpoint["sample"]

        self.trainer = Trainer(
            self.language_model,
            optimizer,
            starting_epoch,
            starting_sample,
            checkpoint_file=checkpoint_file,
        )

    def train(self, epochs: int, batch_size: int):
        collate = Collate(self.tokenizer, device=self.device)

        train_ds = TinyStoriesDS(f"train")
        valid_ds = TinyStoriesDS("validation")

        train_dl = DataLoader(
            train_ds,
            batch_size,
            collate_fn=collate.collate,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

        valid_dl = DataLoader(
            valid_ds,
            batch_size,
            collate_fn=collate.collate,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

        self.trainer.fit(train_dl, valid_dl, epochs)

    def test(self, batch_size: int):
        collate = Collate(self.tokenizer, device=self.device)
        valid_ds = TinyStoriesDS("valid")
        valid_dl = DataLoader(
            valid_ds, batch_size, collate_fn=collate.collate, shuffle=True
        )

        self.trainer.test(valid_dl)

    def predict(self, str: str, max_length: int) -> Optional[str]:
        return self.language_model.predict(str, max_length, beam_size=5)

    def model_summary(self):
        summary(
            self.language_model,
            input_size=(
                10,  # batch size
                20,  # sequence length
            ),
            dtypes=[torch.int64],
            device=self.device,
            depth=10,
        )
