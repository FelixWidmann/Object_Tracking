import torch
import torch.nn as nn
from .transformer import TransformerEncoderLayer, TransformerDecoderLayer

class HypothesisGenerator(nn.Module):
    def __init__(self, d=256, heads=8, num_queries=300, enc_layers=4, dec_layers=4):
        super().__init__()
        self.num_queries = num_queries

        self.encoder = nn.ModuleList([TransformerEncoderLayer(d, heads) for _ in range(enc_layers)])
        self.decoder = nn.ModuleList([TransformerDecoderLayer(d, heads) for _ in range(dec_layers)])

        self.proposal_queries = nn.Parameter(torch.randn(num_queries, d))

        self.bbox_head = nn.Linear(d, 4)
        self.obj_head  = nn.Linear(d, 1)

    def forward(self, img_feat):
        B, HW, d = img_feat.shape

        # Encode
        x = img_feat
        for layer in self.encoder:
            x = layer(x)

        # Decode proposals
        q = self.proposal_queries.unsqueeze(0).repeat(B,1,1)
        for layer in self.decoder:
            q = layer(q, x)

        bbox = self.bbox_head(q).sigmoid()
        obj  = self.obj_head(q).sigmoid()

        return q, bbox, obj
