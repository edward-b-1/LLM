import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

from config import ModelConfig
from model.gpt import GPT
from data.dataset import TokenDataset


@dataclass
class TrainConfig:
    # Data
    train_path: str = "datasets/shakespeare_train.bin"
    val_path:   str = "datasets/shakespeare_val.bin"

    # Optimizer — swap this to experiment
    # Options: "sgd" | "sgd_momentum" | "adamw"
    optimizer:    str   = "adamw"
    lr:           float = 3e-4
    momentum:     float = 0.9    # only used by sgd_momentum
    weight_decay: float = 0.1    # only used by adamw

    # Training
    batch_size: int   = 16
    max_steps:  int   = 100_000
    grad_clip:  float = 1.0    # set to 0.0 to disable

    # Logging & checkpointing
    log_interval:  int = 10
    eval_interval: int = 250
    eval_steps:    int = 20
    checkpoint_dir: str = "checkpoints"


def get_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def find_latest_checkpoint(checkpoint_dir, optimizer):
    pattern = os.path.join(checkpoint_dir, f"{optimizer}_step*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.split("step")[-1].replace(".pt", "")))


def load_checkpoint(path, model, optimizer):
    print(f"Resuming from {path}")
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"] + 1


@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(model_cfg=None, train_cfg=None, fresh=False):
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Optimizer : {train_cfg.optimizer}  lr={train_cfg.lr}")
    print(f"Model     : {model_cfg.n_layers}L  d={model_cfg.d_model}  heads={model_cfg.n_heads}")

    # Data
    train_ds = TokenDataset(train_cfg.train_path, model_cfg.context_length)
    val_ds   = TokenDataset(train_cfg.val_path,   model_cfg.context_length)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model & optimizer
    model = GPT(model_cfg).to(device)
    optimizer = get_optimizer(model, train_cfg)
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # Resume from latest checkpoint unless --fresh was requested
    step = 0
    if not fresh:
        ckpt_path = find_latest_checkpoint(train_cfg.checkpoint_dir, train_cfg.optimizer)
        if ckpt_path:
            step = load_checkpoint(ckpt_path, model, optimizer)
        else:
            print("No checkpoint found — starting from scratch.")
    else:
        print("--fresh specified — starting from scratch.")

    print(f"Parameters  : {model.num_params():,}")
    print(f"Train tokens: {len(train_ds):,} samples")
    print(f"Starting at : step {step}")
    print("-" * 60)

    loader_iter = iter(train_loader)

    while step < train_cfg.max_steps:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        if train_cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.log_interval == 0:
            print(f"step {step:5d} | train loss {loss.item():.4f}")

        is_eval_step = step > 0 and step % train_cfg.eval_interval == 0
        is_last_step = step == train_cfg.max_steps - 1

        if is_eval_step or is_last_step:
            val_loss = evaluate(model, val_loader, train_cfg.eval_steps, device)
            print(f"{'':>8} | val loss   {val_loss:.4f}  ← step {step}")

            path = os.path.join(train_cfg.checkpoint_dir,
                                f"{train_cfg.optimizer}_step{step}.pt")
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_cfg": train_cfg,
                "model_cfg": model_cfg,
            }, path)
            print(f"{'':>8} | saved {path}")

        step += 1

    print("-" * 60)
    print("Training complete.")


DATASETS = {
    "shakespeare":          ("datasets/shakespeare_train.bin",          "datasets/shakespeare_val.bin"),
    "shakespeare_complete": ("datasets/shakespeare_complete_train.bin", "datasets/shakespeare_complete_val.bin"),
    "wikipedia":            ("datasets/wikipedia_train.bin",            "datasets/wikipedia_val.bin"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh",      action="store_true",
                        help="Ignore existing checkpoints and train from scratch")
    parser.add_argument("--dataset",    type=str,   default=None,
                        choices=list(DATASETS.keys()),
                        help="Dataset to train on (default: shakespeare)")
    parser.add_argument("--optimizer",  type=str,   default=None,
                        choices=["sgd", "sgd_momentum", "adamw"],
                        help="Optimizer to use (default: from TrainConfig)")
    parser.add_argument("--max-steps",  type=int,   default=None,
                        help="Number of training steps (default: from TrainConfig)")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Learning rate (default: from TrainConfig)")
    parser.add_argument("--grad-clip",  type=float, default=None,
                        help="Gradient clipping max norm, 0 to disable (default: 1.0)")
    args = parser.parse_args()

    train_cfg = TrainConfig()
    if args.dataset is not None:
        train_cfg.train_path, train_cfg.val_path = DATASETS[args.dataset]
    if args.optimizer is not None:
        train_cfg.optimizer = args.optimizer
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.grad_clip is not None:
        train_cfg.grad_clip = args.grad_clip

    train(train_cfg=train_cfg, fresh=args.fresh)
