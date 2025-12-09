import torch
import torch.nn as nn
from .transformer import TransformerDecoderLayer

class MemoryDecoder(nn.Module):
    def __init__(self, d=256, heads=8, dec_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d, heads) for _ in range(dec_layers)])

        self.bbox_head = nn.Linear(d, 4)
        self.obj_head  = nn.Linear(d, 1)
        self.uni_head  = nn.Linear(d, 1)

    def forward(self, queries, img_feat):
        x = queries
        for layer in self.layers:
            x = layer(x, img_feat)

        bbox = self.bbox_head(x).sigmoid()
        obj  = self.obj_head(x).sigmoid()
        uni  = self.uni_head(x).sigmoid()

        confidence = obj * uni     # Eq. (1)

        return x, bbox, obj, uni, confidence
