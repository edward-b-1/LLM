from dataclasses import dataclass


@dataclass
class SFTConfig:
    # Optimizer
    lr:           float = 1e-5
    weight_decay: float = 0.1
    lr_min:       float = 0.0
    warmup_steps: int   = 100

    # Training
    batch_size: int   = 16
    max_steps:  int   = 10_000
    grad_clip:  float = 1.0

    # Logging & checkpointing
    log_interval:   int = 10
    eval_interval:  int = 250
    eval_steps:     int = 50
    save_interval:  int = 2_000
    checkpoint_dir: str = "checkpoints"
    log_dir:        str = "logs"
    run_name:       str = "sft"
