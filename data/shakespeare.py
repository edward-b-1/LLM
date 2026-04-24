import urllib.request
import os
import numpy as np
from data.prepare import get_tokenizer, save, DATASET_DIR

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare():
    raw_path = os.path.join(DATASET_DIR, "shakespeare.txt")

    if not os.path.exists(raw_path):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(URL, raw_path)

    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Raw text: {len(text):,} characters")

    tokenizer = get_tokenizer()
    tokens = tokenizer(text, return_attention_mask=False)["input_ids"]

    print(f"Tokens: {len(tokens):,}")
    save(tokens, "shakespeare")


if __name__ == "__main__":
    prepare()
