import array
import os
import numpy as np
from datasets import load_dataset
from data.prepare import get_tokenizer, DATASET_DIR
from config import ModelConfig

BUFFER_TOKENS  = 10_000_000   # flush to disk every ~20MB
COPY_CHUNK_MB  = 100          # MB per chunk when splitting train/val
VAL_FRACTION   = 0.005
CONTEXT_LENGTH = ModelConfig().context_length


def prepare():
    tokenizer = get_tokenizer()
    eos = tokenizer.eos_token_id

    print("Loading Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    n = len(ds)
    print(f"Articles: {n:,}")

    tmp_path = os.path.join(DATASET_DIR, "wikipedia_all.bin")
    buffer = array.array('H')
    total_tokens = 0
    article_offsets = []

    with open(tmp_path, 'wb') as f:
        for i, article in enumerate(ds):
            article_offsets.append(total_tokens + len(buffer))
            ids = tokenizer(
                f"{article['title']}\n\n{article['text']}",
                return_attention_mask=False,
                truncation=False,
            )["input_ids"]
            ids.append(eos)
            buffer.extend(ids)

            if len(buffer) >= BUFFER_TOKENS:
                buffer.tofile(f)
                total_tokens += len(buffer)
                buffer = array.array('H')

            if i % 10_000 == 0:
                print(f"  {i:,} / {n:,} articles — {total_tokens + len(buffer):,} tokens")

        if buffer:
            buffer.tofile(f)
            total_tokens += len(buffer)

    print(f"\nTotal tokens: {total_tokens:,}")
    _split(tmp_path, total_tokens, article_offsets)
    os.remove(tmp_path)
    print("Done.")


def _split(src_path, total_tokens, article_offsets):
    split_token = int((1 - VAL_FRACTION) * total_tokens)
    split_byte  = split_token * 2
    assert split_byte % 2 == 0, "split_byte must be uint16-aligned"
    chunk_bytes = COPY_CHUNK_MB * 1024 * 1024

    train_path = os.path.join(DATASET_DIR, "wikipedia_train.bin")
    val_path   = os.path.join(DATASET_DIR, "wikipedia_val.bin")

    print(f"Writing train ({split_token:,} tokens)...")
    with open(src_path, 'rb') as src, open(train_path, 'wb') as dst:
        remaining = split_byte
        while remaining > 0:
            chunk = src.read(min(chunk_bytes, remaining))
            dst.write(chunk)
            remaining -= len(chunk)

    print(f"Writing val ({total_tokens - split_token:,} tokens)...")
    with open(src_path, 'rb') as src, open(val_path, 'wb') as dst:
        src.seek(split_byte)
        while chunk := src.read(chunk_bytes):
            dst.write(chunk)

    # Train offsets: articles that start within train and have a full context window
    train_offsets = np.array(
        [o for o in article_offsets if o + CONTEXT_LENGTH + 1 <= split_token],
        dtype=np.uint64,
    )

    # Val offsets: articles that start within val and have a full context window,
    # rebased to be relative to the start of the val file
    val_total = total_tokens - split_token
    val_offsets = np.array(
        [o - split_token for o in article_offsets
         if o >= split_token and o - split_token + CONTEXT_LENGTH + 1 <= val_total],
        dtype=np.uint64,
    )

    train_offsets_path = os.path.join(DATASET_DIR, "wikipedia_train_offsets.npy")
    val_offsets_path   = os.path.join(DATASET_DIR, "wikipedia_val_offsets.npy")
    np.save(train_offsets_path, train_offsets)
    np.save(val_offsets_path,   val_offsets)

    print(f"Train -> {train_path}  ({len(train_offsets):,} articles)")
    print(f"Val   -> {val_path}  ({len(val_offsets):,} articles)")


if __name__ == "__main__":
    prepare()
