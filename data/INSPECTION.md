# Data Inspection

Run these in an IPython session from the project root (`llm/`).

## 1. Dataset structure

```python
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
print(ds)               # dataset size, column names, format
print(ds.features)      # column types
```

## 2. Individual articles

```python
article = ds[0]
print(article.keys())           # e.g. id, url, title, text

print(article['title'])
print(article['text'][:1000])   # first 1000 characters
```

## 3. Article length statistics

```python
import numpy as np

sample = ds.select(range(5000))
char_lengths = [len(a['text']) for a in sample]

print(f"Mean length : {np.mean(char_lengths):,.0f} chars")
print(f"Median      : {np.median(char_lengths):,.0f} chars")
print(f"Min         : {min(char_lengths):,} chars")
print(f"Max         : {max(char_lengths):,} chars")
```

## 4. Tokenised binary files

```python
import numpy as np
from transformers import GPT2TokenizerFast

tokens = np.memmap("datasets/shakespeare_train.bin", dtype=np.uint16, mode='r')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

print(f"Total tokens : {len(tokens):,}")
print(f"First 20 IDs : {tokens[:20]}")
print()
print("Decoded first 200 tokens:")
print(tokenizer.decode(tokens[:200]))
```

## 5. EOS boundaries in Wikipedia

```python
import numpy as np

tokens = np.memmap("datasets/wikipedia_train.bin", dtype=np.uint16, mode='r')
eos_positions = np.where(tokens == 50256)[0]

print(f"Total tokens      : {len(tokens):,}")
print(f"Article boundaries: {len(eos_positions):,}")
print(f"Mean article len  : {np.mean(np.diff(eos_positions)):,.0f} tokens")
print(f"First boundary at : token {eos_positions[0]:,}")
```
