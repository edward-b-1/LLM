import argparse
import numpy as np
from data.prepare import get_tokenizer, DATASET_DIR
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="train", choices=["train", "val"])
    parser.add_argument("--n",      type=int, default=5, help="Number of examples to show")
    parser.add_argument("--offset", type=int, default=0, help="Start from this index")
    args = parser.parse_args()

    tokens_path = os.path.join(DATASET_DIR, f"squad_{args.split}_tokens.npy")
    masks_path  = os.path.join(DATASET_DIR, f"squad_{args.split}_masks.npy")

    examples = np.load(tokens_path, allow_pickle=True)
    masks    = np.load(masks_path,  allow_pickle=True)
    tokenizer = get_tokenizer()

    print(f"Split: {args.split}  |  Total examples: {len(examples):,}")
    print("=" * 60)

    for i in range(args.offset, min(args.offset + args.n, len(examples))):
        tokens = examples[i]
        mask   = masks[i]

        question_tokens = [t for t, m in zip(tokens, mask) if m == 0]
        answer_tokens   = [t for t, m in zip(tokens, mask) if m == 1]

        question = tokenizer.decode(question_tokens)
        answer   = tokenizer.decode(answer_tokens)

        print(f"[{i}] {question.strip()}")
        print(f"     {answer.strip()}")
        print()


if __name__ == "__main__":
    main()
