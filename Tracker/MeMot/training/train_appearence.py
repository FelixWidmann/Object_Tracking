#!/usr/bin/env python3
"""
Train a lightweight appearance embedding network
Outputs: models/appearance_head.pth
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import ReIDDataset
from losses import TripletLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "models/appearance_head.pth"
EMB_DIM = 128


class AppearanceHead(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(512, EMB_DIM)

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)


def train():
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor()
    ])

    train_data = ReIDDataset("data/train", transform=transform)
    loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

    model = AppearanceHead().to(DEVICE)
    loss_fn = TripletLoss(margin=0.3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(30):
        for imgs, ids in loader:
            imgs = imgs.to(DEVICE)
            ids = ids.to(DEVICE)

            emb = model(imgs)
            loss = loss_fn(emb, ids)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print("Saved:", SAVE_PATH)


if __name__ == "__main__":
    train()
