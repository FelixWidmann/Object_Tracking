import torch
import torch.nn as nn
from .aggregator import MemoryAggregator

class MemoryEncoder(nn.Module):
    def __init__(self, d=256, heads=8, Ts=3, Tl=24):
        super().__init__()
        self.agg = MemoryAggregator(d, heads, Ts, Tl)

    def forward(self, memory, dmat):
        return self.agg(memory, dmat)
