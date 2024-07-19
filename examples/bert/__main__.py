import os

from absl import app as absl_app
import torch
import typer
from typing_extensions import Annotated

from examples.bert.bert_example import BERTExample

app = typer.Typer(no_args_is_help=True, add_completion=False)


def bert_example(device: torch.device = None):
    return BERTExample(checkpoint_file="examples/bert/checkpoint.pt", device=device)


@app.command()
def summary():
    """A summary of the model using torchinfo."""
    example = bert_example()
    example.model_summary()


@app.command()
def train(
    epochs: Annotated[int, typer.Argument()],
    batch_size: Annotated[int, typer.Argument()],
    device: Annotated[str, typer.Option(help="Can be cpu or cuda.")] = None,
):
    """Train the model."""

    print(f"Training the model with a batch size of {batch_size} for {epochs} epochs.")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    example = bert_example(device)
    example.train(epochs, batch_size)


from transformers import AutoTokenizer
from examples.bert.bookcorpus import BookCorpus


@app.command()
def tokenizer():
    # tokenizer = AutoTokenizer.from_pretrained("arlette-tokenizer32")
    train_ds = BookCorpus()

    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = old_tokenizer.train_new_from_iterator(train_ds, 32000)
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
            "mask_token": "[MSK]",
        }
    )
    tokenizer.save_pretrained("tokenizer")


@app.command()
def dataset():
    example = bert_example()
    example.create_dataset()


if __name__ == "__main__":
    app()
