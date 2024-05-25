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
from examples.translator.text_pairs import TextPairs
from torch import device, randperm
from transformers import AutoTokenizer
import torch
from pathlib import Path
from examples.translator.translator import Translator
from typing import Optional

# from . import Collate
from typing import Iterable, cast, Any


class LMDataset(Dataset):
    """Loader for MultiNLI development set."""

    def __init__(self, tokenizer: Any):
        super().__init__()
        train_ds = TextPairs()
        train_perm: list[int] = randperm(len(train_ds)).tolist()
        train_ds_sample = [train_ds[i] for i in train_perm[:100]]

        #src = "I'm pretty busy here."
        #tgt = "Je suis plutôt occupé ici."
        #train_ds_sample2 = [(src, tgt)]

        self._examples = [
            {
                "src_text": src,
                "tgt_text": tokenizer.bos_token + tgt,
                "ref_text": tgt + tokenizer.eos_token,
            }
            for src, tgt in train_ds_sample
        ]

    def spec(self) -> Spec:
        return {
            "src_text": TextSegment(),
            "tgt_text": TextSegment(),
            "ref_text": TextSegment(),
        }


class TranslatorExample(Model):
    def __init__(self, tokenizer: Any, checkpoint_file: Optional[str] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device("cpu")
        self.model = Translator(tokenizer, device=self.device)
        if checkpoint_file is not None and Path(checkpoint_file).is_file():
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def predict(self, inputs: Iterable[JsonDict], **kw) -> Iterable[JsonDict]:

        src_token_ids = [self.tokenizer.encode(x["src_text"]) for x in inputs]
        tgt_token_ids = [self.tokenizer.encode(x["tgt_text"]) for x in inputs]

        src_tokens = [self.tokenizer.tokenize(x["src_text"]) for x in inputs]
        tgt_tokens = [self.tokenizer.tokenize(x["tgt_text"]) for x in inputs]
        ref_tokens = [self.tokenizer.tokenize(x["ref_text"]) for x in inputs]

        batch_size = len(src_token_ids)
        src_max_length = len(max(src_token_ids, key=lambda x: len(x)))
        tgt_max_length = len(max(tgt_token_ids, key=lambda x: len(x)))

        src_token_ids_batch: torch.Tensor = torch.full(
            (batch_size, src_max_length),
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.int64,
        )

        tgt_token_ids_batch: torch.Tensor = torch.full(
            (batch_size, tgt_max_length),
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.int64,
        )

        for batch, (src, tgt) in enumerate(zip(src_token_ids, tgt_token_ids)):
            src_token_ids_batch[batch, : len(src)] = torch.tensor(
                src,
                device=self.device,
            )
            tgt_token_ids_batch[batch, : len(tgt)] = torch.tensor(
                tgt,
                device=self.device,
            )

        outputs: list[JsonDict] = []

        self.model.eval()
        preds, attn_weights = self.model.forward_norm(
            src_token_ids_batch, tgt_token_ids_batch, average_weights=False
        )

        for batch, (
            embeddings,
            enc_attn_weights,
            dec_self_attn_weights,
            dec_cross_attn_weights,
        ) in enumerate(
            zip(
                preds,
                attn_weights["enc_attn_weights"],
                attn_weights["dec_self_attn_weights"],
                attn_weights["dec_cross_attn_weights"],
            )
        ):
            # embeddings = softmax(embeddings, dim=-1)
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
                    "src_tokens": src_tokens[batch],
                    "tgt_tokens": tgt_tokens[batch],
                    "ref_tokens": ref_tokens[batch],
                    "pred_tokens": acc,
                    "embeddings": embeddings[0, :].cpu().detach().numpy(),
                }
                | {
                    "enc/layer_"
                    + str(i)
                    + "/attention": enc_attn_weights[i].cpu().detach().numpy()
                    for i in range(self.model.num_layers)
                }
                | {
                    "dec/layer_"
                    + str(i)
                    + "/self_attention": dec_self_attn_weights[i].cpu().detach().numpy()
                    for i in range(self.model.num_layers)
                }
                | {
                    "dec/layer_"
                    + str(i)
                    + "/cross_attention": dec_cross_attn_weights[i]
                    .cpu()
                    .detach()
                    .numpy()
                    for i in range(self.model.num_layers)
                }
            )

        return outputs

    def input_spec(self) -> Spec:
        return {
            "src_text": TextSegment(),
            "tgt_text": TextSegment(),
            "ref_text": TextSegment(),
        }

    def output_spec(self) -> Spec:
        return (
            {
                "output_text": GeneratedText(parent="ref_text"),
                "src_tokens": Tokens(parent="src_text"),
                "tgt_tokens": Tokens(parent="tgt_text"),
                "ref_tokens": Tokens(parent="ref_text"),
                "pred_tokens": TokenTopKPreds(align="ref_tokens"),
                "embeddings": Embeddings(),
            }
            | {
                "enc/layer_"
                + str(i)
                + "/attention": AttentionHeads(
                    align_in="src_tokens", align_out="src_tokens"
                )
                for i in range(self.model.num_layers)
            }
            | {
                "dec/layer_"
                + str(i)
                + "/self_attention": AttentionHeads(
                    align_in="tgt_tokens", align_out="tgt_tokens"
                )
                for i in range(self.model.num_layers)
            }
            | {
                "dec/layer_"
                + str(i)
                + "/cross_attention": AttentionHeads(
                    align_in="src_tokens", align_out="tgt_tokens"
                )
                for i in range(self.model.num_layers)
            }
        )




def start(_):
    tokenizer = AutoTokenizer.from_pretrained("examples/translator/tokenizer")

    datasets = {
        "dataset": LMDataset(tokenizer),
    }

    models = {
        "model": TranslatorExample(
            tokenizer, checkpoint_file="examples/translator/model.pt"
        ),
    }

    lit_demo = Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()
