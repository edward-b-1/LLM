import argparse
import glob
import os
import torch
from data.prepare import get_tokenizer

from config import ModelConfig
from model.gpt import GPT
from train import TrainConfig  # required to unpickle checkpoints


def find_latest_checkpoint(checkpoint_dir, optimizer=None):
    pattern = os.path.join(checkpoint_dir, f"{optimizer}_step*.pt" if optimizer else "*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}" +
                                (f" for optimizer '{optimizer}'" if optimizer else ""))
    return max(candidates, key=lambda p: int(p.split("step")[-1].replace(".pt", "")))


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_cfg = ckpt["model_cfg"]
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    tokens = tokenizer(prompt, return_attention_mask=False)["input_ids"]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    context_length = model.cfg.context_length

    for _ in range(max_new_tokens):
        x_cond = x[:, -context_length:]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(x_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            logits[logits < top_values[:, -1:]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",         type=str,   default="",
                        help="Text prompt to continue")
    parser.add_argument("--checkpoint",     type=str,   default=None,
                        help="Path to a specific checkpoint .pt file")
    parser.add_argument("--optimizer",      type=str,   default=None,
                        choices=["sgd", "sgd-momentum", "adamw"],
                        help="Load latest checkpoint for this optimizer")
    parser.add_argument("--run-name",       type=str,   default=None,
                        help="Load latest checkpoint for this run name")
    parser.add_argument("--max-new-tokens", type=int,   default=200,
                        help="Number of tokens to generate (default: 200)")
    parser.add_argument("--temperature",    type=float, default=0.8,
                        help="Sampling temperature: lower=focused, higher=creative (default: 0.8)")
    parser.add_argument("--top-k",          type=int,   default=50,
                        help="Sample from top-k tokens only (default: 50)")
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
    print("-" * 60)

    output = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(output)
