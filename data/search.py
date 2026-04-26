import argparse
import numpy as np
from data.prepare import get_tokenizer, DATASET_DIR
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query",       type=str,
                        help="Text to search for")
    parser.add_argument("--dataset",   type=str, default="wikipedia")
    parser.add_argument("--split",     type=str, default="train", choices=["train", "val"])
    parser.add_argument("--max-results", type=int, default=10,
                        help="Maximum number of matches to return (default: 10)")
    parser.add_argument("--context",   type=int, default=128,
                        help="Tokens of context to show around each match (default: 128)")
    args = parser.parse_args()

    bin_path     = os.path.join(DATASET_DIR, f"{args.dataset}_{args.split}.bin")
    offsets_path = os.path.join(DATASET_DIR, f"{args.dataset}_{args.split}_offsets.npy")

    tokenizer = get_tokenizer()
    tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')

    query_ids = tokenizer(args.query, return_attention_mask=False)["input_ids"]
    query_arr = np.array(query_ids, dtype=np.uint16)
    n = len(query_arr)

    print(f"Searching for: {repr(args.query)}  ({n} tokens: {query_ids})")

    if os.path.exists(offsets_path):
        # Search only at article boundaries
        offsets = np.load(offsets_path)
        matches = []
        for offset in offsets:
            offset = int(offset)
            chunk = tokens[offset : offset + n]
            if len(chunk) == n and np.array_equal(chunk, query_arr):
                matches.append(offset)
                if len(matches) >= args.max_results:
                    break
        print(f"Searched {len(offsets):,} article offsets")
    else:
        # Full scan
        matches = []
        for i in range(len(tokens) - n + 1):
            if np.array_equal(tokens[i : i + n], query_arr):
                matches.append(i)
                if len(matches) >= args.max_results:
                    break
        print(f"Scanned {len(tokens):,} tokens")

    print(f"Found {len(matches)} match(es)\n")

    for match in matches:
        start = max(0, match - args.context)
        end   = min(len(tokens), match + n + args.context)
        text  = tokenizer.decode(tokens[start:end].tolist())
        print(f"--- offset {match} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
