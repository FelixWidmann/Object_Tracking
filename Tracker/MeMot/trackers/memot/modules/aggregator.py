import torch
import torch.nn as nn
from .transformer import TransformerDecoderLayer

class MemoryAggregator(nn.Module):
    def __init__(self, d, heads, Ts, Tl):
        super().__init__()
        self.Ts = Ts
        self.Tl = Tl

        self.short_attn = TransformerDecoderLayer(d, heads)   # f_short
        self.long_attn  = TransformerDecoderLayer(d, heads)   # f_long
        self.fusion     = TransformerDecoderLayer(d, heads)   # f_fusion

    def forward(self, memory, dmat):
        """
        memory: (N, T, d)
        dmat:   (N, d)
        """

        N, T, d = memory.shape

        # Short-term memory
        short_mem = memory[:, -self.Ts:, :]    # (N, Ts, d)
        short_q   = memory[:, -1:, :]          # latest observation
        ast       = self.short_attn(short_q, short_mem).squeeze(1)

        # Long-term memory
        long_q = dmat.unsqueeze(1)            # (N,1,d)
        alt = self.long_attn(long_q, memory).squeeze(1)

        # Fuse
        fused = torch.stack([ast, alt], dim=1)   # (N,2,d)
        fused = self.fusion(fused, fused).mean(1)

        # Updated DMAT (Eq. in paper)
        new_dmat = fused

        return fused, new_dmat
