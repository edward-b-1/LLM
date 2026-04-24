import array
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from data.prepare import save

N_WORKERS  = 16    # match physical core count
CHUNK_SIZE = 500   # articles per job dispatched to each worker

# Worker state — initialised once per process, not once per article
_tokenizer = None

def _init_worker():
    global _tokenizer
    from transformers import GPT2TokenizerFast
    _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    _tokenizer.model_max_length = 1_000_000

def _tokenize_article(text):
    ids = _tokenizer(text, return_attention_mask=False, truncation=False)["input_ids"]
    ids.append(_tokenizer.eos_token_id)
    return ids


def prepare():
    print("Loading Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    n = len(ds)
    print(f"Articles: {n:,}  |  Workers: {N_WORKERS}  |  Chunk size: {CHUNK_SIZE}")

    # array.array stores uint16 at 2 bytes/token — avoids Python list overhead (~28 bytes/int)
    all_tokens = array.array('H')

    # Generator streams one article at a time — avoids loading all text into RAM upfront
    article_texts = (article["text"] for article in ds)

    with mp.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        for i, ids in enumerate(pool.imap(_tokenize_article, article_texts, chunksize=CHUNK_SIZE)):
            all_tokens.extend(ids)
            if i % 50_000 == 0:
                print(f"  {i:,} / {n:,} articles — {len(all_tokens):,} tokens")

    print(f"\nTotal tokens: {len(all_tokens):,}")

    tokens = np.frombuffer(all_tokens, dtype=np.uint16).copy()
    save(tokens, "wikipedia")


if __name__ == "__main__":
    prepare()
