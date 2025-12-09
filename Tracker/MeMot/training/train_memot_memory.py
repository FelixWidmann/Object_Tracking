#!/usr/bin/env python3
"""
Fine-tune MeMOT memory module (ΘE, ΘD, DMAT)
Outputs: models/memot_memory.pth
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import MOTDataset
from losses import HungarianAssociationLoss
from trackers.memot.memot import MeMOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "models/memot_memory.pth"


def train():
    dataset = MOTDataset("data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = MeMOT(Ts=3, Tl=24).to(DEVICE)

    # Freeze everything except the memory modules
    for name, param in model.named_parameters():
        if not any(k in name for k in ["E", "D", "dmat"]):
            param.requires_grad = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    loss_fn = HungarianAssociationLoss()

    for epoch in range(30):
        for batch in loader:
            img_feat, det_xywh, det_emb, gt_assign = batch

            img_feat = img_feat.to(DEVICE)
            det_xywh = det_xywh.to(DEVICE)
            det_emb = det_emb.to(DEVICE)
            gt_assign = gt_assign.to(DEVICE)

            # Forward
            N = det_xywh.shape[1]
            q = torch.zeros(1, N, model.d, device=DEVICE)
            dec_out, _, _, _, _ = model.D(q, img_feat)

            loss = loss_fn(dec_out.squeeze(0), det_emb.squeeze(0), gt_assign)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print("Saved:", SAVE_PATH)


if __name__ == "__main__":
    train()
