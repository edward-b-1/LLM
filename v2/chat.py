import argparse
import torch
from data.prepare import get_tokenizer

from generate import find_latest_checkpoint, load_model, generate
from train import TrainConfig   # required to unpickle checkpoints
from sft import SFTConfig       # required to unpickle SFT checkpoints

PROMPT_PREFIX = "Question: "
PROMPT_SUFFIX = "\nAnswer: "


def strip_prompt(output, prompt):
    if output.startswith(prompt):
        return output[len(prompt):]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     type=str,   default=None)
    parser.add_argument("--optimizer",      type=str,   default=None,
                        choices=["sgd", "sgd-momentum", "adamw"])
    parser.add_argument("--run-name",       type=str,   default=None)
    parser.add_argument("--max-new-tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top-k",          type=int,   default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        name = args.run_name or args.optimizer
        ckpt_path = find_latest_checkpoint("checkpoints", name)

    print(f"Loading: {ckpt_path}")
    model, model_cfg = load_model(ckpt_path, device)
    tokenizer = get_tokenizer()

    print(f"Model : {model_cfg.n_layers}L  d={model_cfg.d_model}  params={model.num_params():,}")
    print(f"Temp  : {args.temperature}  Top-k: {args.top_k}")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        prompt = PROMPT_PREFIX + question + PROMPT_SUFFIX

        output = generate(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        answer = strip_prompt(output, prompt)
        print(f"Bot: {answer}")


if __name__ == "__main__":
    main()
