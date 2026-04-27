import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from model.gpt import GPT
from data.dataset import TokenDataset
from data.registry import VAL_DATASETS
from train import TrainConfig, evaluate
from sft import SFTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--dataset",    type=str, default="wikipedia",
                        choices=list(VAL_DATASETS.keys()))
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="Number of batches to evaluate over (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model_cfg = ckpt["model_cfg"]
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Model  : {model_cfg.n_layers}L  d={model_cfg.d_model}  params={model.num_params():,}")
    print(f"Dataset: {args.dataset}")

    val_ds     = TokenDataset(VAL_DATASETS[args.dataset], model_cfg.context_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    val_loss = evaluate(model, val_loader, args.eval_steps, device)
    print(f"Val loss: {val_loss:.4f}  (perplexity {torch.exp(torch.tensor(val_loss)):.2f})")


if __name__ == "__main__":
    main()
