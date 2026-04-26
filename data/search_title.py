import argparse
import numpy as np
from datasets import load_dataset
from data.prepare import get_tokenizer, DATASET_DIR
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query",     type=str, help="Article title to search for")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--split",   type=str, default="train", choices=["train", "val"])
    parser.add_argument("--length",  type=int, default=256,
                        help="Tokens to display from the matching article (default: 256)")
    parser.add_argument("--exact",   action="store_true",
                        help="Require exact title match (default: case-insensitive substring)")
    args = parser.parse_args()

    offsets_path = os.path.join(DATASET_DIR, f"{args.dataset}_{args.split}_offsets.npy")
    bin_path     = os.path.join(DATASET_DIR, f"{args.dataset}_{args.split}.bin")

    if not os.path.exists(offsets_path):
        raise FileNotFoundError(f"Offsets file not found: {offsets_path} — re-run data preparation")

    offsets = np.load(offsets_path)

    print("Loading Wikipedia metadata...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    query_lower = args.query.lower()
    match_index = None
    match_title = None

    for i, article in enumerate(ds):
        title = article["title"]
        if args.exact:
            hit = title == args.query
        else:
            hit = query_lower in title.lower()
        if hit:
            match_index = i
            match_title = title
            break

    if match_index is None:
        print(f"No article found matching '{args.query}'")
        return

    if match_index >= len(offsets):
        print(f"Article {match_index} ('{match_title}') is in the val split, not train")
        return

    offset = int(offsets[match_index])
    tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
    tokenizer = get_tokenizer()

    chunk = tokens[offset : offset + args.length]
    print(f"Title   : '{match_title}'")
    print(f"Article : {match_index:,}")
    print(f"Offset  : {offset:,}")
    print(f"Tokens  : {offset} – {offset + len(chunk) - 1}  (length {len(chunk)})")
    print("-" * 60)
    print(tokenizer.decode(chunk.tolist()))


if __name__ == "__main__":
    main()
