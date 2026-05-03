import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.d_model * cfg.ffn_mult
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)
