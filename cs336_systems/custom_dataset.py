from torch import tensor, long
from torch.utils.data import Dataset

class DecoderDataset(Dataset):
    def __init__(self, rows, char_to_idx, max_length):
        self.rows = rows
        self.char_to_idx = char_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text = self.rows[index][:self.max_length]
        tokens = tensor(
            [self.char_to_idx[c] for c in text],
            dtype=long,
        )
        return tokens[:-1], tokens[1:]

