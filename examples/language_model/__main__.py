from absl import app as absl_app
from examples.language_model.lm_example import LMExample
import typer
from typing_extensions import Annotated
from examples.language_model.lit import start
import torch
import os

app = typer.Typer(no_args_is_help=True, add_completion=False)


def lm_example(device: torch.device = None):
    return LMExample(
        checkpoint_file="examples/language_model/checkpoint.pt", device=device
    )


@app.command()
def summary():
    """A summary of the model using torchinfo."""
    example = lm_example()
    example.model_summary()


@app.command()
def generate(
    prefix: Annotated[str, typer.Option()] = "Once upon a time",
    max_length: Annotated[int, typer.Option()] = 50,
    device: Annotated[str, typer.Option(help="Can be cpu or cuda.")] = "cpu",
):
    """Generate a text starting with a given prefix."""
    example = lm_example(device)
    print(example.predict(prefix, max_length))


@app.command()
def train(
    epochs: Annotated[int, typer.Argument()],
    batch_size: Annotated[int, typer.Argument()],
    device: Annotated[str, typer.Option(help="Can be cpu or cuda.")] = None,
):
    """Train the model."""

    print(f"Training the model with a batch size of {batch_size} for {epochs} epochs.")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    example = lm_example(device)
    example.train(epochs, batch_size)

@app.command()
def test(
    batch_size: Annotated[int, typer.Argument()],
    device: Annotated[str, typer.Option(help="Can be cpu or cuda.")] = None,
):
    """Test the model on the validation set."""

    print(f"Testing the model on the validation set with a batch size of {batch_size}.")
    example = lm_example(device)
    example.test(batch_size)


@app.command()
def lit():
    """Load the model with the Learning Interpretability Tool (LIT)."""
    absl_app.run(start)


if __name__ == "__main__":
    app()
