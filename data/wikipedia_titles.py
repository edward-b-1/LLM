from datasets import load_dataset
import os
from data.prepare import DATASET_DIR


def extract_titles():
    print("Loading Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    print(f"Articles: {len(ds):,}")

    out_path = os.path.join(DATASET_DIR, "wikipedia_titles.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for article in ds:
            f.write(article["title"] + "\n")

    print(f"Written to {out_path}")


if __name__ == "__main__":
    extract_titles()
