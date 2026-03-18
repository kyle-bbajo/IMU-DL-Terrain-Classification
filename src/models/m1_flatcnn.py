# -*- coding: utf-8 -*-
"""
src/models/flatcnn.py — FlatCNN (베이스라인)
입력: 로우 신호 54ch × 256pt → flatten → 1D CNN
"""
from __future__ import annotations
import torch.nn as nn
from .blocks import ConvBNReLU, _init
from ..utils.config import CFG


class FlatCNN(nn.Module):
    """
    베이스라인 모델.
    신호를 그룹 분리 없이 flat하게 CNN에 입력.
    입력: (B, C, T) — C=54, T=256
    """
    def __init__(self, in_ch: int, num_classes: int = CFG.num_classes):
        super().__init__()
        d = CFG.dropout_feat
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(64, 128, 5, 2),   nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(128, 256, 3, 1),  nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, CFG.feat_dim), nn.ReLU(inplace=True), nn.Dropout(CFG.dropout_clf),
        )
        self.head = nn.Linear(CFG.feat_dim, num_classes)
        self.apply(_init)

    def forward(self, x):
        return self.head(self.net(x))

    def extract(self, x):
        return self.net(x)
