import os
import numpy as np
from transformers import GPT2TokenizerFast

DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(DATA_DIR), "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = 1_000_000
    return tokenizer


def save(tokens, name, val_fraction=0.005):
    split = int((1 - val_fraction) * len(tokens))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    train_path = os.path.join(DATASET_DIR, f"{name}_train.bin")
    val_path   = os.path.join(DATASET_DIR, f"{name}_val.bin")

    np.array(train_tokens, dtype=np.uint16).tofile(train_path)
    np.array(val_tokens,   dtype=np.uint16).tofile(val_path)

    print(f"Train: {len(train_tokens):,} tokens -> {train_path}")
    print(f"Val:   {len(val_tokens):,} tokens  -> {val_path}")
