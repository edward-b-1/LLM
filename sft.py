import os
import csv
import glob
import math
import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from model.gpt import GPT
from data.dataset import SquadDataset
from train import get_lr, find_latest_checkpoint, TrainConfig
from configs import SFTConfig


def load_pretrained(path, model):
    print(f"Loading pre-trained weights from {path}")
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt.get("model_cfg", ModelConfig())


def load_sft_checkpoint(path, model, optimizer):
    print(f"Resuming SFT from {path}")
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"] + 1


@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device):
    model.eval()
    losses = []
    for i, (x, y, mask) in enumerate(val_loader):
        if i >= eval_steps:
            break
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y, loss_mask=mask)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(sft_cfg=None, pretrained_checkpoint=None, fresh=False):
    sft_cfg = sft_cfg or SFTConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Run      : {sft_cfg.run_name}")

    train_ds = SquadDataset("train", ModelConfig().context_length)
    val_ds   = SquadDataset("val",   ModelConfig().context_length)

    train_loader = DataLoader(train_ds, batch_size=sft_cfg.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=sft_cfg.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model_cfg = ModelConfig()
    model     = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=sft_cfg.lr, weight_decay=sft_cfg.weight_decay)

    os.makedirs(sft_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(sft_cfg.log_dir, exist_ok=True)
    log_path = os.path.join(sft_cfg.log_dir, f"{sft_cfg.run_name}.csv")

    step = 1
    sft_ckpt = find_latest_checkpoint(sft_cfg.checkpoint_dir, sft_cfg.run_name)

    if not fresh and sft_ckpt:
        step = load_sft_checkpoint(sft_ckpt, model, optimizer)
    elif pretrained_checkpoint:
        load_pretrained(pretrained_checkpoint, model)
    else:
        raise ValueError("Provide --pretrained <path> to start SFT from a pre-trained checkpoint, "
                         "or --fresh to train from random initialisation.")

    print(f"Parameters : {model.num_params():,}")
    print(f"Train examples: {len(train_ds):,}")
    print(f"Starting at: step {step}")
    print("-" * 60)

    log_mode   = "a" if not fresh and os.path.exists(log_path) else "w"
    log_file   = open(log_path, log_mode, newline="")
    log_writer = csv.writer(log_file)
    if log_mode == "w":
        log_writer.writerow(["step", "train_loss", "val_loss", "lr"])

    train_cfg_proxy = type("cfg", (), {
        "warmup_steps": sft_cfg.warmup_steps,
        "max_steps":    sft_cfg.max_steps,
        "lr":           sft_cfg.lr,
        "lr_min":       sft_cfg.lr_min,
    })()

    loader_iter = iter(train_loader)

    while step <= sft_cfg.max_steps:
        try:
            x, y, mask = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y, mask = next(loader_iter)

        x, y, mask = x.to(device), y.to(device), mask.to(device)

        lr = get_lr(step, train_cfg_proxy)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y, loss_mask=mask)

        optimizer.zero_grad()
        loss.backward()
        if sft_cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip)
        optimizer.step()

        if step % sft_cfg.log_interval == 0:
            print(f"step {step:5d} | train loss {loss.item():.4f}  lr {lr:.2e}")

        is_eval_step = step % sft_cfg.eval_interval == 0
        is_last_step = step == sft_cfg.max_steps
        is_save_step = sft_cfg.save_interval > 0 and step % sft_cfg.save_interval == 0

        val_loss = None
        if is_eval_step or is_last_step:
            val_loss = evaluate(model, val_loader, sft_cfg.eval_steps, device)
            print(f"{'':>8} | val loss   {val_loss:.4f}  ← step {step}")

        log_writer.writerow([step, f"{loss.item():.4f}",
                             f"{val_loss:.4f}" if val_loss is not None else "",
                             f"{lr:.2e}"])
        log_file.flush()

        if is_save_step or is_last_step:
            path = os.path.join(sft_cfg.checkpoint_dir, f"{sft_cfg.run_name}_step{step}.pt")
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sft_cfg":   sft_cfg,
                "model_cfg": model_cfg,
            }, path)
            print(f"{'':>8} | saved {path}")

        step += 1

    log_file.close()
    print("-" * 60)
    print("SFT complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained",    type=str, default=None,
                        help="Path to pre-trained checkpoint to start SFT from")
    parser.add_argument("--fresh",         action="store_true",
                        help="Ignore existing SFT checkpoints and restart")
    parser.add_argument("--run-name",      type=str, default=None)
    parser.add_argument("--max-steps",     type=int, default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--batch-size",    type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    args = parser.parse_args()

    sft_cfg = SFTConfig()
    if args.run_name      is not None: sft_cfg.run_name      = args.run_name
    if args.max_steps     is not None: sft_cfg.max_steps     = args.max_steps
    if args.lr            is not None: sft_cfg.lr            = args.lr
    if args.batch_size    is not None: sft_cfg.batch_size    = args.batch_size
    if args.save_interval is not None: sft_cfg.save_interval = args.save_interval

    train(sft_cfg=sft_cfg, pretrained_checkpoint=args.pretrained, fresh=args.fresh)
