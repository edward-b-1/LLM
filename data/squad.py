import os
import numpy as np
from datasets import load_dataset
from data.prepare import get_tokenizer, DATASET_DIR


PROMPT_TEMPLATE = "Question: {question}\nAnswer: {answer}"


def prepare():
    tokenizer = get_tokenizer()

    print("Loading SQuAD...")
    train_ds = load_dataset("rajpurkar/squad", split="train")
    val_ds   = load_dataset("rajpurkar/squad", split="validation")
    print(f"Train examples: {len(train_ds):,}")
    print(f"Val examples  : {len(val_ds):,}")

    for split_name, ds in [("train", train_ds), ("val", val_ds)]:
        tokens_list  = []
        masks_list   = []

        for example in ds:
            question = example["question"].strip()
            answer   = example["answers"]["text"][0].strip()

            question_ids = tokenizer(
                f"Question: {question}\nAnswer: ",
                return_attention_mask=False,
            )["input_ids"]

            answer_ids = tokenizer(
                answer,
                return_attention_mask=False,
            )["input_ids"]
            answer_ids.append(tokenizer.eos_token_id)

            ids  = question_ids + answer_ids
            mask = [0] * len(question_ids) + [1] * len(answer_ids)

            tokens_list.append(ids)
            masks_list.append(mask)

        tokens_path = os.path.join(DATASET_DIR, f"squad_{split_name}_tokens.npy")
        masks_path  = os.path.join(DATASET_DIR, f"squad_{split_name}_masks.npy")

        np.save(tokens_path, np.array(tokens_list, dtype=object))
        np.save(masks_path,  np.array(masks_list,  dtype=object))

        print(f"{split_name}: {len(tokens_list):,} examples -> {tokens_path}")


if __name__ == "__main__":
    prepare()
