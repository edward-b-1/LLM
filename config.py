from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size:     int   = 50257   # GPT-2 BPE vocabulary
    context_length: int   = 1024    # fixed sequence length
    d_model:        int   = 512     # embedding / hidden dimension
    n_heads:        int   = 8       # attention heads (head_dim = d_model / n_heads = 64)
    n_layers:       int   = 8       # transformer blocks
    ffn_mult:       int   = 4       # FFN hidden = ffn_mult * d_model
    dropout:        float = 0.1
