import absl
import torch
from examples.translator.translator_example import TranslatorExample
import typer
from typing_extensions import Annotated

app = typer.Typer(no_args_is_help=True, add_completion=False)


def translator_example(device: torch.device = None):
    return TranslatorExample(
        checkpoint_file="examples/translator/checkpoint.pt", device=device
    )


@app.command()
def summary():
    """A summary of the model using torchinfo."""
    example = translator_example()
    example.model_summary()


@app.command()
def translate(
    text: Annotated[str, typer.Argument()],
    device: Annotated[str, typer.Option(help="Can be cpu or cuda.")] = None,
):
    """Generate a text starting with a given prefix."""
    example = translator_example(device)
    print(example.translator.predict(text))


@app.command()
def train(
    epochs: Annotated[int, typer.Argument()],
    batch_size: Annotated[int, typer.Argument()],
    device: Annotated[str, typer.Option(help="Can be cpu or cuda.")] = None,
):
    """Train the model."""

    print(f"Training the model with a batch size of {batch_size} for {epochs} epochs.")
    example = translator_example(device)
    example.train(epochs, batch_size)


@app.command()
def lit():
    """Load the model with the Learning Interpretability Tool (LIT)."""
    absl.app.run(start)


if __name__ == "__main__":
    app()
