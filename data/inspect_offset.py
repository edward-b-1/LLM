import argparse
import numpy as np
from data.prepare import get_tokenizer, DATASET_DIR
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str, default="wikipedia",
                        help="Dataset name (default: wikipedia)")
    parser.add_argument("--split",    type=str, default="train", choices=["train", "val"])
    parser.add_argument("--offset",   type=int, default=0,
                        help="Token offset to start from (default: 0)")
    parser.add_argument("--article",  type=int, default=None,
                        help="Article index into the offsets file (alternative to --offset)")
    parser.add_argument("--length",   type=int, default=1024,
                        help="Number of tokens to display (default: 1024)")
    args = parser.parse_args()

    bin_path     = os.path.join(DATASET_DIR, f"{args.dataset}_{args.split}.bin")
    offsets_path = os.path.join(DATASET_DIR, f"{args.dataset}_{args.split}_offsets.npy")

    tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
    tokenizer = get_tokenizer()

    offset = args.offset
    if args.article is not None:
        if not os.path.exists(offsets_path):
            raise FileNotFoundError(f"No offsets file found at {offsets_path}")
        offsets = np.load(offsets_path)
        offset = int(offsets[args.article])
        print(f"Article {args.article} starts at token offset {offset}")

    chunk = tokens[offset : offset + args.length]
    print(f"Tokens {offset} – {offset + len(chunk) - 1}  (length {len(chunk)})")
    print("-" * 60)
    print(tokenizer.decode(chunk.tolist()))


if __name__ == "__main__":
    main()
