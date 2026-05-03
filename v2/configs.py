from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    train_path: str = "datasets/shakespeare_train.bin"
    val_path:   str = "datasets/shakespeare_val.bin"

    # Optimizer — swap this to experiment
    # Options: "sgd" | "sgd-momentum" | "adamw"
    optimizer:    str   = "adamw"
    lr:           float = 3e-4
    momentum:     float = 0.9    # only used by sgd-momentum
    weight_decay: float = 0.1    # only used by adamw

    # LR schedule — cosine decay with linear warmup
    lr_min:       float = 0.0
    warmup_steps: int   = 1_000

    # Training
    batch_size:       int   = 1
    grad_accum_steps: int   = 16   # effective batch = batch_size * grad_accum_steps
    max_steps:        int   = 100_000
    grad_clip:        float = 1.0

    # Logging & checkpointing
    log_interval:     int = 10
    eval_interval:    int = 250
    eval_steps:       int = 20
    save_interval:    int = 10_000  # permanent checkpoint every N steps; all versions kept
    rolling_interval: int = 0       # rolling checkpoint every N steps; only latest kept (0 = disabled)
    checkpoint_dir:   str = "checkpoints"
    log_dir:          str = "logs"
    run_name:         str = ""


@dataclass
class SFTConfig:
    # Optimizer
    lr:           float = 1e-5
    weight_decay: float = 0.1
    lr_min:       float = 0.0
    warmup_steps: int   = 100

    # Training
    batch_size:       int   = 1
    grad_accum_steps: int   = 16
    max_steps:        int   = 10_000
    grad_clip:        float = 1.0

    # Logging & checkpointing
    log_interval:   int = 10
    eval_interval:  int = 250
    eval_steps:     int = 50
    save_interval:  int = 2_000
    checkpoint_dir: str = "checkpoints"
    log_dir:        str = "logs"
    run_name:       str = "sft"
