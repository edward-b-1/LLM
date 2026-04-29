import torch
import torch.nn as nn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RoPE(nn.Module):
    def __init__(self, head_dim, context_length, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(context_length).float()
        freqs = torch.outer(t, inv_freq)         # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        self.register_buffer("cos", emb.cos())   # (T, head_dim)
        self.register_buffer("sin", emb.sin())   # (T, head_dim)

    def forward(self, x):
        # x: (B, n_heads, T, head_dim)
        T = x.shape[2]
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        return x * cos + rotate_half(x) * sin
