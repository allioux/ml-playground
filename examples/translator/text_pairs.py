import re

class TextPairs:
    def __init__(self):
        with open('examples/translator/data', 'r') as f:
            self.data: list[tuple[str, str]] = []
            for line in f:
                eng, fr, _, _ = re.split("[\t\n]", line)
                self.data.append((eng, fr))

    def __getitem__(self, item: int) -> tuple[str, str]:
        return self.data[item]

    def __len__(self):
        return len(self.data)