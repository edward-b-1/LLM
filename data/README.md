# Data Pipeline

## Structure

```
data/               Python package — all code lives here
    prepare.py      Shared utilities: tokenizer, save(), paths
    dataset.py      PyTorch Dataset class for training
    shakespeare.py  TinyShakespeare preparation
    wikipedia.py    English Wikipedia preparation (memory-efficient)
    wikipedia_parallel.py  Wikipedia preparation (multiprocess, faster)

datasets/           Binary token files — gitignored, never commit
    shakespeare_train.bin
    shakespeare_val.bin
    wikipedia_train.bin
    wikipedia_val.bin
```

## File Format

All `.bin` files are raw binary sequences of `uint16` token IDs (2 bytes each).
No headers, no metadata — just packed integers. Read with `numpy.memmap`.

Token IDs are from the GPT-2 BPE vocabulary (50,257 tokens).
Articles are separated by the EOS token (ID 50256).

## Preparing Datasets

```bash
# TinyShakespeare (~1MB text, seconds to prepare)
python -m data.shakespeare

# Wikipedia, single-threaded, memory-efficient (~hours)
python -m data.wikipedia

# Wikipedia, 16-process parallel (~faster)
python -m data.wikipedia_parallel
```

Output files are written to `datasets/` relative to the project root.

## Using in Training

```python
from data.dataset import TokenDataset
from torch.utils.data import DataLoader

train_ds = TokenDataset("datasets/wikipedia_train.bin", context_length=1024)
loader   = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

for x, y in loader:
    # x: (32, 1024) int64 — input token IDs
    # y: (32, 1024) int64 — target token IDs (x shifted right by 1)
    ...
```
