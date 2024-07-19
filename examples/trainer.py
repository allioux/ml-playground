import sys
import time
from typing import TypeVar, Optional, cast, Sized, TypedDict

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict
from torch.utils.data import DataLoader
from termcolor import colored

from examples.model import Model

In = TypeVar("In", contravariant=True, bound=Sized)
Out = TypeVar("Out", contravariant=True, bound=Sized)


class Checkpoint(TypedDict):
    epoch: int
    sample: int
    model_state_dict: StateDict
    optimizer_state_dict: StateDict


class Trainer:
    def __init__(
        self,
        model: Model[In, Out],
        optimizer: Optimizer,
        starting_epoch: int = 0,
        starting_sample: int = 0,
        clip_grad_norm: Optional[float] = None,
        checkpoint_file: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.starting_epoch = starting_epoch
        self.starting_sample = starting_sample
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_file = checkpoint_file

    def train(self, dataloader: DataLoader[In], epoch: int) -> None:
        size = len(dataloader.dataset)
        sample = self.starting_sample
        update_time = time.time()
        checkpoint_time = time.time()

        batch_size: int = dataloader.batch_size
        start_batch = self.starting_sample // batch_size
        sample_offset = self.starting_sample % batch_size

        print(f"EPOCH {epoch}")
        refresh = False
        # The typing information of the dataloader is lost when iterating over it!
        # (PyTorch issue #119123)
        for batch, (x, y) in enumerate(dataloader):

            # Not optimal…
            if batch < start_batch:
                if (
                    time.time() - update_time > 1
                    or batch == 0
                    or batch == start_batch - 1
                ):
                    update_time = time.time()
                    sys.stdout.write(
                        colored(
                            f"\rSkipping batches: {int(batch / start_batch * 100)}%",
                            "magenta",
                        )
                    )
                    sys.stdout.flush()
                continue

            if batch == start_batch:
                x = x[sample_offset:]
                y = y[sample_offset:]

            sample = min((batch + 1) * batch_size, size)

            if time.time() - update_time > 5 or sample == size or batch == start_batch:
                update_time = time.time()
                refresh = True
                training_results = self.model.training_step(x, y, with_accuracy=True)
            else:
                training_results = self.model.training_step(x, y, with_accuracy=False)

            loss = training_results["loss"]
            accuracy = training_results["accuracy"]
            loss.backward()

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if refresh:
                refresh = False
                percent = int(100 * sample / size)
                bar = "=" * int(60 * sample / size)
                progress = (
                    f"\r[{bar:<60}] Training {percent}% "
                    f"| sample: {sample}/{size} "
                    f"| loss: {loss:.3} "
                )
                progress += f"| accuracy: {accuracy:.3}" if accuracy is not None else ""
                sys.stdout.write(colored(progress, "magenta"))
                sys.stdout.flush()

            if self.checkpoint_file is not None and time.time() - checkpoint_time > 30:
                checkpoint_time = time.time()
                checkpoint: Checkpoint = {
                    "epoch": epoch,
                    "sample": sample,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }

                torch.save(
                    checkpoint,
                    self.checkpoint_file,
                )

            self.starting_sample = sample

        sys.stdout.write("\n")
        sys.stdout.flush()

    def test(self, test_dl: DataLoader[In]) -> None:
        size = len(test_dl.dataset)
        sample = 0
        loss, accuracy = 0.0, 0.0

        with torch.no_grad():
            # The typing information of the dataloader is lost when iterating over it!
            # (PyTorch issue #119123)
            for x, y in test_dl:
                sample += len(y)

                test_results = self.model.test_step(x, y)
                ratio = len(y) / size
                loss += test_results["loss"].item() * ratio
                accuracy += cast(float, test_results["accuracy"]) * ratio
                avg_loss = loss * size / sample
                avg_accuracy = accuracy * size / sample

                percent = int(100 * sample / size)
                bar = "=" * int(60 * sample / size)
                progress = (
                    f"\r[{bar:<60}] Validation {percent}% "
                    f"| sample: {sample}/{size} "
                    f"| avg. loss: {avg_loss:.3} "
                    f"| avg. accuracy: {avg_accuracy:.3}"
                )
                sys.stdout.write(colored(progress, "cyan"))
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

    def fit(
        self,
        train_dl: DataLoader[In],
        test_dl: Optional[DataLoader[In]] = None,
        epochs: int = 1,
    ) -> None:
        starting_epoch = self.starting_epoch
        for epoch in range(starting_epoch, starting_epoch + epochs):
            #self.test(test_dl)
            self.train(train_dl, epoch)
            #if test_dl is not None:
            #    self.test(test_dl)
            self.starting_epoch += 1
            self.starting_sample = 0
