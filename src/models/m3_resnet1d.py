# -*- coding: utf-8 -*-
"""src/models/resnet1d.py — ResNet1D (비교 모델 2)"""
from __future__ import annotations
import torch, torch.nn as nn
from .blocks import ConvBNReLU, ResBlock1D, CrossGroupAttn, _init
from ..utils.config import CFG


class _ResNetBranch(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        d = CFG.dropout_feat
        self.stem   = ConvBNReLU(in_ch, 64, 7, 3)
        self.layer1 = nn.Sequential(ResBlock1D(64, 64, 1, 7, "none"), ResBlock1D(64, 64, 1, 7, "cbam"))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, 2, 5, "none"), ResBlock1D(128, 128, 1, 5, "cbam"))
        self.layer3 = nn.Sequential(ResBlock1D(128, out_dim, 2, 3, "none"), ResBlock1D(out_dim, out_dim, 1, 3, "cbam"))
        self.drop   = nn.Dropout(d)

    def forward(self, x):
        return self.drop(self.layer3(self.layer2(self.layer1(self.stem(x)))))


class ResNet1D(nn.Module):
    """비교 모델 2 — 잔차 블록 기반 1D CNN"""
    def __init__(self, branch_ch: dict[str, int], num_classes: int = CFG.num_classes):
        super().__init__()
        feat = CFG.feat_dim; dc = CFG.dropout_clf
        self.names    = list(branch_ch.keys())
        self.branches = nn.ModuleDict({nm: _ResNetBranch(ch, feat) for nm, ch in branch_ch.items()})
        self.cross    = CrossGroupAttn(feat)
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.clf      = nn.Sequential(
            nn.Linear(feat * len(self.names), 256), nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(256, 128),                    nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(128, num_classes),
        )
        self.apply(_init)

    def _encode(self, bi):
        feats = [self.pool(self.branches[nm](bi[nm])).squeeze(-1) for nm in self.names]
        x     = self.cross(torch.stack(feats, dim=1))
        return torch.cat([x[:, i, :] for i in range(len(self.names))], dim=1)

    def forward(self, bi): return self.clf(self._encode(bi))
    def extract(self, bi): return self._encode(bi)
