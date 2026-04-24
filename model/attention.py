import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # Project input to Q, K, V in one shot — cheaper than three separate linears
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)                          # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, n_heads, T, head_dim) for batched attention
        def to_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # is_causal=True applies the causal mask without materialising an N×N matrix
        # PyTorch dispatches to Flash Attention kernels here when available
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads: (B, n_heads, T, head_dim) -> (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)
