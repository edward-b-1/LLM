import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: lm_head and tok_emb share the same matrix.
        # Intuition: the same geometry that maps tokens→vectors should map vectors→token scores.
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"Sequence length {T} exceeds context_length {self.cfg.context_length}"

        x = self.drop(self.tok_emb(idx))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            if loss_mask is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def num_params(self):
        # Subtract lm_head params since they're shared with tok_emb
        return sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()
