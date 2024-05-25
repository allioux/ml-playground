from lit_nlp import server_flags
from lit_nlp.api.dataset import Dataset
from lit_nlp.api.model import Model
from lit_nlp.api.types import (
    Spec,
    TextSegment,
    GeneratedText,
    JsonDict,
    Tokens,
    Embeddings,
    AttentionHeads,
    TokenTopKPreds,
)
from lit_nlp.dev_server import Server
from examples.language_model.tiny_stories import TinyStoriesDS
from torch import device, randperm
from torch.utils.data import Subset
from torch.nn.functional import softmax
from transformers import AutoTokenizer
import torch
from pathlib import Path

# from . import Collate
from examples.language_model.transformer_lm import TransformerLM
from typing import Iterable, Optional, cast, Any

class LMDataset(Dataset):
    """Loader for MultiNLI development set."""

    def __init__(self, tokenizer: Any):
        super().__init__()
        train_ds = TinyStoriesDS("train")
        train_perm: list[int] = randperm(len(train_ds)).tolist()
        train_ds_sample = Subset(train_ds, train_perm[:10])

        self._examples = [
            {
                "input_text": tokenizer.bos_token + text,
                "target_text": text + tokenizer.eos_token,
            }
            for text in train_ds_sample
        ]

    def spec(self) -> Spec:
        return {"input_text": TextSegment(), "target_text": TextSegment()}


class LMModel(Model):
    def __init__(self, tokenizer: Any, checkpoint_file: Optional[str] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device("cpu")
        self.model = TransformerLM(
            self.tokenizer, 256, 1024, device=self.device, dropout=0.1
        )
        if checkpoint_file is not None and Path(checkpoint_file).is_file():
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def predict(self, inputs: Iterable[JsonDict], **kw) -> Iterable[JsonDict]:

        input_token_ids = [self.tokenizer.encode(x["input_text"]) for x in inputs]
        input_tokens = [self.tokenizer.tokenize(x["input_text"]) for x in inputs]
        target_tokens = [self.tokenizer.tokenize(x["target_text"]) for x in inputs]

        batch_size = len(input_token_ids)
        max_length = len(max(input_token_ids, key=lambda x: len(x)))

        token_ids_batch: torch.Tensor = torch.empty(
            (batch_size, max_length), device=self.device, dtype=torch.int64
        )

        for batch, seq in enumerate(input_token_ids):
            length = len(seq)
            pad_length = max_length - length
            token_ids_batch[batch] = torch.tensor(
                seq + pad_length * [self.tokenizer.pad_token_id],
                device=self.device,
            )

        outputs: list[JsonDict] = []

        preds = self.model(token_ids_batch, average_weights=False)
        preds = preds[0], preds[1] if preds[1] is not None else len(preds) * [None]

        for batch, (embeddings, attn_weights) in enumerate(zip(*preds)):
            embeddings = softmax(embeddings, dim=-1)
            out_tokens = self.tokenizer.convert_ids_to_tokens(
                cast(list[int], torch.argmax(embeddings, dim=-1).tolist())
            )
            preds = torch.topk(embeddings, 10, dim=-1)
            acc = []
            for probs, indices in zip(*preds):
                tokens = self.tokenizer.convert_ids_to_tokens(
                    cast(list[int], indices.tolist())
                )
                acc.append(list(zip(tokens, cast(list[float], probs.tolist()))))

            output_text = self.tokenizer.convert_tokens_to_string(out_tokens)

            outputs.append(
                {
                    "output_text": output_text,
                    "input_tokens": input_tokens[batch],
                    "target_tokens": target_tokens[batch],
                    "pred_tokens": acc,
                    "embeddings": embeddings[0, :].cpu().detach().numpy(),
                }
                | (
                    {
                        "layer_"
                        + str(i)
                        + "/attention": attn_weights[i].cpu().detach().numpy()
                        for i in range(6)
                    }
                    if attn_weights is not None
                    else {}
                )
            )

        return outputs

    def input_spec(self) -> Spec:
        return {
            "input_text": TextSegment(),
            "target_text": TextSegment(),
        }

    def output_spec(self) -> Spec:
        return {
            "output_text": GeneratedText(parent="target_text"),
            "input_tokens": Tokens(parent="input_text"),
            "pred_tokens": TokenTopKPreds(align="target_tokens"),
            "target_tokens": Tokens(parent="target_text"),
            "embeddings": Embeddings(),
        } | {
            "layer_"
            + str(i)
            + "/attention": AttentionHeads(
                align_in="target_tokens", align_out="target_tokens"
            )
            for i in range(6)
        }


def start(_):
    tokenizer = AutoTokenizer.from_pretrained("examples/language_model/tokenizer")

    datasets = {
        "dataset": LMDataset(tokenizer),
    }

    models = {
        "model": LMModel(tokenizer, checkpoint_file="examples/language_model/checkpoint.pt"),
    }

    lit_demo = Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()