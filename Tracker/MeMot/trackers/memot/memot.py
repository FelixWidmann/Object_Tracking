import torch
import torch.nn as nn

# Correct imports for module files
from .modules.hypothesis import HypothesisGenerator
from .modules.memory_encoder import MemoryEncoder
from .modules.memory_decoder import MemoryDecoder
from .modules.aggregator import MemoryAggregator
from .modules.transformer import TransformerEncoderLayer

# ======================
# Bounding box utilities
# ======================

def bbox_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return [cx, cy, w, h]

def xywh_to_bbox(xywh):
    cx, cy, w, h = xywh
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]

class MeMOT(nn.Module):
    def __init__(self,
                 d=256,
                 heads=8,
                 Ts=3,
                 Tl=24,
                 num_queries=300):
        super().__init__()

        self.Tl = Tl
        self.d = d

        self.H = HypothesisGenerator(d, heads, num_queries)
        self.E = MemoryEncoder(d, heads, Ts, Tl)
        self.D = MemoryDecoder(d, heads)

        self.max_tracks = num_queries

        self.register_buffer("memory", torch.zeros(self.max_tracks, Tl, d))
        self.register_buffer("dmat", torch.zeros(self.max_tracks, d))
        self.track_active = torch.zeros(self.max_tracks).bool()

    def forward(self, img_feat):
        B = img_feat.size(0)
        assert B == 1, "Online MOT runs one frame at a time."

        pro_q, pro_bbox, pro_obj = self.H(img_feat)

        active_idx = torch.where(self.track_active)[0]
        track_embeds = self.dmat[active_idx].unsqueeze(0)

        queries = torch.cat([track_embeds, pro_q], dim=1)

        _, bbox, obj, uni, conf = self.D(queries, img_feat)

        return bbox, obj, uni, conf
