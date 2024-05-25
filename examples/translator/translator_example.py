import torch
from transformers import AutoTokenizer
from torch import device, Tensor
from examples.translator.translator import Translator
from pathlib import Path
from examples.translator.text_pairs import TextPairs
from typing import Optional, Any
from ml_playground.utils import TorchKw
from examples.translator.translator import Translator
from torch.utils.data import DataLoader
from examples.trainer import Trainer
from torchinfo import summary
import transformers


class Collate:
    def __init__(
        self,
        tokenizer: Any,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
        device: Optional[device] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = "cpu"  # device
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def collate(self, samples: list[str]) -> tuple[tuple[Tensor, Tensor], Tensor]:
        src_token_ids: list[list[int]]
        tgt_token_ids: list[list[int]]
        src_token_ids, tgt_token_ids = [], []

        pad_token_id: int = self.tokenizer.pad_token_id
        bos_token_id: int = self.tokenizer.bos_token_id
        eos_token_id: int = self.tokenizer.eos_token_id

        for src, tgt in samples:
            src_token_ids.append(self.tokenizer.encode(src))
            tgt_token_ids.append(self.tokenizer.encode(tgt))

        batch_size = len(src_token_ids)

        if self.max_src_length is None:
            max_src_length = len(max(src_token_ids, key=lambda x: len(x))) + 1
        else:
            max_src_length = self.max_src_length

        if self.max_tgt_length is None:
            max_tgt_length = len(max(tgt_token_ids, key=lambda x: len(x))) + 1
        else:
            max_tgt_length = self.max_tgt_length

        kwargs: TorchKw = {"dtype": torch.int64, "device": self.device}

        src_ids = torch.full((batch_size, max_src_length), pad_token_id, **kwargs)
        tgt_ids = torch.full((batch_size, max_tgt_length), pad_token_id, **kwargs)
        y_ids = torch.full((batch_size, max_tgt_length), pad_token_id, **kwargs)

        for batch, (s, t) in enumerate(zip(src_token_ids, tgt_token_ids)):
            src_length = min(max_src_length, len(s))
            src_ids[batch, :src_length] = torch.tensor(s[:src_length], **kwargs)

            tgt_length = min(max_tgt_length - 1, len(t))
            tgt_ids[batch, : tgt_length + 1] = torch.tensor(
                [bos_token_id] + t[:tgt_length], **kwargs
            )

            y_ids[batch, : tgt_length + 1] = torch.tensor(
                t[:tgt_length] + [eos_token_id], **kwargs
            )

        return (src_ids, tgt_ids), y_ids


class TranslatorExample:
    def __init__(
        self, checkpoint_file: Optional[str] = None, device: torch.device = None
    ) -> None:
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
        self.tokenizer = AutoTokenizer.from_pretrained("examples/translator/tokenizer")
        self.translator = Translator(self.tokenizer, dropout=0.2, device=self.device)
        optimizer = torch.optim.Adam(self.translator.parameters())
        starting_epoch = 0
        starting_sample = 0

        if checkpoint_file is not None and Path(checkpoint_file).is_file():
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.translator.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            starting_epoch = checkpoint["epoch"]
            starting_sample = checkpoint["sample"]

        self.trainer = Trainer(
            self.translator,
            optimizer,
            starting_epoch,
            starting_sample,
            checkpoint_file=checkpoint_file,
        )

    def train(self, epochs: int, batch_size: int):
        data = TextPairs()

        collate = Collate(self.tokenizer, device=self.device)

        train_size = 0.9
        test_start = int(train_size * len(data))

        train_dl = DataLoader(
            data[:test_start],
            batch_size,
            collate_fn=collate.collate,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_dl = DataLoader(
            data[test_start:],
            batch_size,
            collate_fn=collate.collate,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.trainer.fit(train_dl, test_dl, epochs)

    def model_summary(self):
        summary(
            self.translator,
            input_size=[
                (
                    10,  # batch size
                    20,  # sequence length
                ),
                (
                    10,  # batch size
                    20,  # sequence length
                ),
            ],
            dtypes=[torch.int64, torch.int64],
            device=self.device,
            depth=10,
        )
