from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from typing import TypeVar, cast

T_co = TypeVar("T_co", covariant=True)


class TinyStoriesDS(Dataset[str]):

    def __init__(self, split: str = "train") -> None:
        ds: DatasetDict = cast(DatasetDict, load_dataset("roneneldan/TinyStories", split=split))
        ds = ds.with_format("torch")
        self.data = ds

    def __getitem__(self, item: int) -> str:
        return self.data[item]["text"]

    def __len__(self):
        return len(self.data)
