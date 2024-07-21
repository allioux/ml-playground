from random import random, seed
from pathlib import Path
from typing import Optional, Any

import numpy.random
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import transformers
from transformers import AutoTokenizer
from torch import device, Tensor
import spacy
import pandas as pd

from examples.bert.bert_model import BERTModel
from examples.trainer import Trainer
from examples.bert.wikipedia import Wikipedia


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
        self.max_length = 512
        self.nlp = spacy.load("en_core_web_sm")
        seed()

    def collate(
        self, samples: list[tuple[str, str]]
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        # samples_token_ids: list[list[int]] = []

        vocab_size = 32000

        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = self.tokenizer.mask_token_id
        sep_token_id = self.tokenizer.sep_token_id
        cls_token_id = self.tokenizer.cls_token_id

        batch_size = len(samples)

        pairs = []
        for sent1, sent2 in samples:
            # extract the first two sentences
            # sents = self.nlp(sample).sents[:2]

            # tokenize the sentences
            sent1_ids = self.tokenizer.encode(sent1)
            sent2_ids = self.tokenizer.encode(sent2)

            pairs.append((sent1_ids, sent2_ids))

        nsp_tgt = torch.empty((batch_size,), dtype=torch.int64, device=self.device)
        for i in range(len(pairs)):
            if random() > 0.5:
                nsp_tgt[i] = 1
            else:
                pairs[i] = (pairs[i][0], pairs[(i + 1) % len(pairs)][1])
                nsp_tgt[i] = 0

        seq_length = max([len(s1) + len(s2) for (s1, s2) in pairs]) + 2

        token_ids = torch.empty(
            (batch_size, seq_length), dtype=torch.int64, device=self.device
        )

        mask = torch.full(
            (batch_size, seq_length), False, dtype=torch.bool, device=self.device
        )

        segments = torch.zeros(
            (batch_size, seq_length), dtype=torch.int64, device=self.device
        )

        for batch, (sent1, sent2) in enumerate(pairs):
            pad_length = seq_length - 2 - len(sent1) - len(sent2)
            token_ids[batch] = torch.tensor(
                [cls_token_id]
                + sent1
                + [sep_token_id]
                + sent2
                + pad_length * [pad_token_id],
                dtype=torch.int64,
                device=self.device,
            )
            mask[batch, 1 : len(sent1) + 1] = True
            mask[batch, len(sent1) + 2 : len(sent1) + 2 + len(sent2)] = True

            segments[batch, len(sent1) + 1 :] = 1

        alter_mask = mask.clone()

        # alter 15% of tokens
        alter_mask[mask] = (
            torch.bernoulli(torch.full(alter_mask[mask].shape, 0.15)) == 1.0
        )

        # 80% of the altered tokens will be masked
        mask_mask = alter_mask.clone()
        foo = mask_mask[alter_mask]
        mask_mask[alter_mask] = (
            torch.bernoulli(torch.full(mask_mask[alter_mask].shape, 0.8)) == 1.0
        )

        # the remaining altered tokens will be substituted with a random one
        mask_rand = torch.logical_xor(alter_mask, mask_mask)

        token_ids[mask_mask] = self.tokenizer.mask_token_id
        token_ids[mask_rand] = (
            torch.rand(token_ids[mask_rand].shape) * (vocab_size - 1) + 1
        ).long()

        mlm_tgt = token_ids[mask_mask]

        return ((token_ids, segments), (mask_mask, mlm_tgt, nsp_tgt))


class BERTExample:
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
        self.tokenizer = AutoTokenizer.from_pretrained("examples/bert/tokenizer")

        self.model = BERTModel(
            self.tokenizer,
            256,
            ff_hidden_dim=1024,
            device=self.device,
            dropout=0.2,
        )

        optimizer = torch.optim.Adam(self.model.parameters())
        starting_epoch = 0
        starting_sample = 0

        if checkpoint_file is not None and Path(checkpoint_file).is_file():
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            starting_epoch = checkpoint["epoch"]
            starting_sample = checkpoint["sample"]

        self.trainer = Trainer(
            self.model,
            optimizer,
            starting_epoch,
            starting_sample,
            checkpoint_file=checkpoint_file,
        )

    def train(self, epochs: int, batch_size: int):
        collate = Collate(self.tokenizer, device=self.device)

        train_ds = Wikipedia()
        valid_ds = Wikipedia()

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
        valid_ds = Wikipedia()
        valid_dl = DataLoader(
            valid_ds, batch_size, collate_fn=collate.collate, shuffle=False
        )

        self.trainer.test(valid_dl)

    def model_summary(self):
        summary(
            self.model,
            input_size=(
                10,  # batch size
                20,  # sequence length
            ),
            dtypes=[torch.int64],
            device=self.device,
            depth=10,
        )
