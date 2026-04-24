import urllib.request
import os
import numpy as np
from data.prepare import get_tokenizer, save, DATASET_DIR

# Complete works of Shakespeare — Project Gutenberg #100
URL = "https://www.gutenberg.org/files/100/100-0.txt"


def prepare():
    raw_path = os.path.join(DATASET_DIR, "shakespeare_complete.txt")

    if not os.path.exists(raw_path):
        print("Downloading complete works of Shakespeare from Project Gutenberg...")
        urllib.request.urlretrieve(URL, raw_path)

    with open(raw_path, "r", encoding="utf-8-sig") as f:
        text = f.read()

    # Strip Gutenberg header and footer boilerplate
    start = text.find("THE COMPLETE WORKS OF WILLIAM SHAKESPEARE")
    end   = text.rfind("End of the Project Gutenberg")
    if start != -1:
        text = text[start:end]

    print(f"Raw text : {len(text):,} characters")

    tokenizer = get_tokenizer()
    tokens = tokenizer(text, return_attention_mask=False)["input_ids"]
    print(f"Tokens   : {len(tokens):,}")

    save(tokens, "shakespeare_complete")


if __name__ == "__main__":
    prepare()
