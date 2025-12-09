import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict, deque

from .memot import MeMOT as MeMOT_Model
from .memot import xywh_to_bbox, bbox_to_xywh


# ============================================================
# Utility: IoU
# ============================================================

def bbox_iou(a, b):
    """ Compute IoU between boxes a and b. """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0

    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


# ============================================================
# Utility: NMS for YOLO outputs (extra safety)
# ============================================================

def nms_boxes(dets, iou_threshold=0.45):
    """Apply NMS manually to remove duplicate boxes."""
    if len(dets) == 0:
        return dets

    boxes = np.array([d["bbox"] for d in dets])
    scores = np.array([d["score"] for d in dets])

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        inds = np.where(iou < iou_threshold)[0]
        order = order[inds + 1]

    return [dets[i] for i in keep]


# ============================================================
# Track object
# ============================================================

class Track:
    def __init__(self, tid, bbox, score, embed, frame_id, max_age=20):
        self.id = tid
        self.bbox = bbox
        self.score = score
        self.embed = embed
        self.last_frame = frame_id
        self.time_since_update = 0
        self.active = True
        self.max_age = max_age

    def update(self, bbox, score, embed, frame_id):
        self.bbox = bbox
        self.score = score
        self.embed = embed
        self.last_frame = frame_id
        self.time_since_update = 0
        self.active = True

    def mark_missed(self):
        self.time_since_update += 1
        if self.time_since_update > self.max_age:
            self.active = False


# ============================================================
# MeMOT Tracker
# ============================================================

class MeMOT_Tracker:

    def __init__(self,
                 mode="yolo",
                 Ts=3,
                 Tl=24,
                 max_age=20,
                 conf_th=0.50,
                 app_th=0.55,
                 device="cuda"):

        self.device = device
        self.mode = mode

        self.max_age = max_age
        self.conf_th = conf_th
        self.app_th = app_th

        # Keep MeMOT model available
        self.model = MeMOT_Model(Ts=Ts, Tl=Tl).to(device)
        self.model.eval()

        # Tracking state
        self.tracks = {}
        self.next_id = 1

        # Appearance embedding extractor
        self.app_pool = nn.AdaptiveAvgPool2d((8, 8)).to(device)
        self.app_dim = 3 * 8 * 8

        # MeMOT memory (unused in YOLO mode)
        self.track_memory = defaultdict(lambda: deque(maxlen=Tl))
        self.track_dmat = {}

        self.backbone = self._placeholder_backbone


    # ============================================================
    # Utility
    # ============================================================

    def _next_id(self):
        tid = self.next_id
        self.next_id += 1
        return tid

    def _placeholder_backbone(self, frame_tensor):
        """Fake feature extractor for MeMOT mode"""
        H, W, d = 20, 20, self.model.d
        return torch.randn(1, H*W, d, device=self.device)


    # ============================================================
    # Appearance embedding (YOLO mode)
    # ============================================================

    def _appearance_embed(self, frame, det_boxes):
        if len(det_boxes) == 0:
            return torch.zeros(0, self.app_dim, device=self.device)

        if isinstance(frame, torch.Tensor):
            frame_np = frame[0].permute(1,2,0).cpu().numpy()
        else:
            frame_np = frame

        H, W = frame_np.shape[:2]
        crops = []

        for b in det_boxes:
            x1,y1,x2,y2 = map(int,b)
            x1=max(0,min(x1,W-1)); x2=max(0,min(x2,W-1))
            y1=max(0,min(y1,H-1)); y2=max(0,min(y2,H-1))

            if x2<=x1 or y2<=y1:
                crop = np.zeros((64,64,3),np.uint8)
            else:
                crop = cv2.resize(frame_np[y1:y2, x1:x2], (64,64))
            crops.append(crop)

        crops = np.stack(crops)
        crops_t = torch.from_numpy(crops).float().to(self.device)/255.0
        crops_t = crops_t.permute(0,3,1,2)

        with torch.no_grad():
            x = self.app_pool(crops_t)
            x = x.reshape(x.size(0), -1)
            x = F.normalize(x, dim=1)

        return x


    # ============================================================
    # YOLO MODE UPDATE + deduplication + IoU gating
    # ============================================================

    def _update_yolo(self, det_boxes, det_scores, det_emb, frame_id):

        # Age tracks
        for tr in self.tracks.values():
            tr.time_since_update += 1

        if len(det_boxes) == 0:
            # age out tracks
            for tid, tr in list(self.tracks.items()):
                tr.mark_missed()
                if not tr.active:
                    del self.tracks[tid]
            return list(self.tracks.values())

        tracks_list = list(self.tracks.values())
        matched_tids = set()
        used_dets = set()

        # ========== Matching via IoU + Appearance ==========
        if len(tracks_list) > 0:
            track_emb = torch.stack([tr.embed for tr in tracks_list])
            track_emb = F.normalize(track_emb, dim=1)
            det_emb_n = F.normalize(det_emb, dim=1)

            sim = (track_emb @ det_emb_n.t()).cpu().numpy()

            for ti, tr in enumerate(tracks_list):
                best_di = None
                best_score = -1

                for di in range(len(det_boxes)):
                    if di in used_dets:
                        continue

                    iou = bbox_iou(tr.bbox, det_boxes[di])
                    if iou < 0.05:
                        continue

                    appearance = sim[ti, di]
                    if appearance > best_score:
                        best_score = appearance
                        best_di = di

                if best_di is not None and best_score >= self.app_th:
                    tr.update(det_boxes[best_di],
                              det_scores[best_di],
                              det_emb[best_di].detach(),
                              frame_id)
                    matched_tids.add(tr.id)
                    used_dets.add(best_di)

        # ========== New Track Birth ==========
        for di in range(len(det_boxes)):
            if di not in used_dets:
                if det_scores[di] >= self.conf_th:
                    tid = self._next_id()
                    tr = Track(tid, det_boxes[di], det_scores[di],
                               det_emb[di].detach(), frame_id,
                               max_age=self.max_age)
                    self.tracks[tid] = tr
                    matched_tids.add(tid)

        # ========== Remove stale tracks ==========
        for tid, tr in list(self.tracks.items()):
            if tid not in matched_tids:
                tr.mark_missed()
                if not tr.active:
                    del self.tracks[tid]

        # ========== REMOVE DUPLICATE TRACKS ==========
        purified = []
        removed = set()
        tlist = list(self.tracks.values())

        for i in range(len(tlist)):
            if tlist[i].id in removed: continue
            for j in range(i+1, len(tlist)):
                if tlist[j].id in removed: continue

                iou = bbox_iou(tlist[i].bbox, tlist[j].bbox)
                if iou > 0.60:          # duplicate track threshold
                    # keep the higher confidence track
                    if tlist[i].score >= tlist[j].score:
                        removed.add(tlist[j].id)
                    else:
                        removed.add(tlist[i].id)

        for tr in tlist:
            if tr.id not in removed:
                purified.append(tr)

        return purified


    # ============================================================
    # Main update
    # ============================================================

    def update(self, detections, frame_id, frame):

        # apply confidence filtering
        detections = [d for d in detections if d["score"] >= self.conf_th]

        # apply extra NMS
        detections = nms_boxes(detections, iou_threshold=0.45)

        det_boxes = [d["bbox"] for d in detections]
        det_scores = [d["score"] for d in detections]

        # YOLO-ONLY MODE
        if self.mode == "yolo":
            det_emb = self._appearance_embed(frame, det_boxes)
            return self._update_yolo(det_boxes, det_scores, det_emb, frame_id)

        # (MeMOT mode left intact)
        return []
