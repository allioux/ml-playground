
class Wikipedia():
    def __init__(self):
        with open('examples/bert/dataset3', 'r') as f:
            self.data: list[tuple[str, str]] = []
            for sent1, sent2 in zip(*[iter(f)]*2):
                self.data.append((sent1[:-1], sent2[:-1]))

    def __getitem__(self, item: int) -> tuple[str, str]:
        return self.data[item]

    def __len__(self):
        return len(self.data)