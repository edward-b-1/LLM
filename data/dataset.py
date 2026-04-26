import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SquadDataset(Dataset):
    def __init__(self, split, context_length):
        from data.prepare import DATASET_DIR
        tokens_path = os.path.join(DATASET_DIR, f"squad_{split}_tokens.npy")
        masks_path  = os.path.join(DATASET_DIR, f"squad_{split}_masks.npy")
        self.examples      = np.load(tokens_path, allow_pickle=True)
        self.masks         = np.load(masks_path,  allow_pickle=True)
        self.context_length = context_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        mask   = self.masks[idx]

        # Truncate to context_length + 1 (need one extra for the shift)
        tokens = tokens[:self.context_length + 1]
        mask   = mask[:self.context_length + 1]

        # Pad to context_length + 1
        pad_len = self.context_length + 1 - len(tokens)
        tokens  = np.array(tokens, dtype=np.int64)
        mask    = np.array(mask,   dtype=np.float32)
        if pad_len > 0:
            tokens = np.concatenate([tokens, np.zeros(pad_len, dtype=np.int64)])
            mask   = np.concatenate([mask,   np.zeros(pad_len, dtype=np.float32)])

        x         = torch.from_numpy(tokens[:-1])
        y         = torch.from_numpy(tokens[1:])
        loss_mask = torch.from_numpy(mask[1:])   # aligned with targets
        return x, y, loss_mask


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
