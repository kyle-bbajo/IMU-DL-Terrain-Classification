# -*- coding: utf-8 -*-
"""
src/models/branchcnn.py — BranchCNN
5그룹 브랜치 CNN + CBAM + CrossGroupAttention + 데이터 증강
입력: bi (dict[str, Tensor]) — 5센서 그룹별 분기 신호
"""
from __future__ import annotations
import torch, torch.nn as nn
from .blocks import ConvBNReLU, CBAM, SEBlock, CrossGroupAttn, FreqBranch, _init
from ..utils.config import CFG


class _Branch(nn.Module):
    def __init__(self, in_ch, out_dim, mode="cbam"):
        super().__init__()
        d = CFG.dropout_feat
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(64, 128, 5, 2),   nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(128, out_dim, 3, 1),
        )
        self.attn = CBAM(out_dim) if mode == "cbam" else SEBlock(out_dim) if mode == "se" else nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.attn(self.net(x))).squeeze(-1)


def _augment(x: torch.Tensor, training: bool) -> torch.Tensor:
    if not training: return x
    b, c, t = x.shape
    if CFG.aug_noise > 0:
        x = x + torch.randn_like(x) * CFG.aug_noise
    if CFG.aug_scale > 0:
        x = x * (1.0 + (torch.rand(b, 1, 1, device=x.device) - 0.5) * CFG.aug_scale)
    if CFG.aug_shift > 0:
        shifts = torch.randint(-CFG.aug_shift, CFG.aug_shift+1, (b,), device=x.device)
        base   = torch.arange(t, device=x.device).view(1, 1, t).expand(b, c, t)
        x      = torch.gather(x, 2, (base - shifts.view(b, 1, 1)) % t)
    return x


class BranchCNN(nn.Module):
    """
    비교 모델 1.
    5센서 그룹 브랜치 CNN + CBAM + CrossGroupAttn + 증강
    """
    def __init__(self, branch_ch: dict[str, int], num_classes: int = CFG.num_classes):
        super().__init__()
        feat = CFG.feat_dim; dc = CFG.dropout_clf
        self.names    = list(branch_ch.keys())
        self.n        = len(self.names)
        self.branches = nn.ModuleDict({nm: _Branch(ch, feat, "cbam") for nm, ch in branch_ch.items()})
        self.cross    = CrossGroupAttn(feat)

        # FFT 브랜치 (Foot 그룹)
        self.use_fft = CFG.use_fft and CFG.fft_group in branch_ch
        if self.use_fft:
            self.fft_src    = CFG.fft_group
            self.freq_branch = _FreqBranch(branch_ch[self.fft_src], feat)

        n_f = self.n + (1 if self.use_fft else 0)
        self.clf = nn.Sequential(
            nn.Linear(feat * n_f, 256), nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(256, 128),        nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(128, num_classes),
        )
        self.apply(_init)

    def _encode(self, bi):
        feats = [self.branches[nm](bi[nm]) for nm in self.names]
        x     = self.cross(torch.stack(feats, dim=1))
        feats = [x[:, i, :] for i in range(self.n)]
        if self.use_fft: feats.append(self.freq_branch(bi[self.fft_src]))
        return torch.cat(feats, dim=1)

    def forward(self, bi):
        if self.training:
            bi = {k: _augment(v, True) for k, v in bi.items()}
        return self.clf(self._encode(bi))

    def extract(self, bi): return self._encode(bi)


class _FreqBranch(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2),
            ConvBNReLU(64, out_dim, 5, 2), nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        return self.net(torch.fft.rfft(x, dim=-1).abs()).squeeze(-1)
