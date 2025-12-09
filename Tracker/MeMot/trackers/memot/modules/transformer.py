import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d)
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = MLP(d)

    def forward(self, x):
        h,_ = self.self_attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ff(x)
        return self.norm2(x + h)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.ff = MLP(d)

    def forward(self, q, kv):
        h,_ = self.self_attn(q, q, q)
        q = self.norm1(q + h)

        h,_ = self.cross_attn(q, kv, kv)
        q = self.norm2(q + h)

        h = self.ff(q)
        return self.norm3(q + h)
