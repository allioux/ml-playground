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


if __name__ == "__main__":
    app()
