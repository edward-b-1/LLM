import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, path, context_length):
        self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
        self.context_length = context_length

        offsets_path = path.replace('.bin', '_offsets.npy')
        self.offsets = np.load(offsets_path) if os.path.exists(offsets_path) else None

    def __len__(self):
        if self.offsets is not None:
            return len(self.offsets)
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        start = int(self.offsets[idx]) if self.offsets is not None else idx * self.context_length
        chunk = self.tokens[start : start + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
