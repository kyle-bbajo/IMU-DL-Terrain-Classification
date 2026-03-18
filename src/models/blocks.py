# -*- coding: utf-8 -*-
"""src/models/blocks.py — 공통 빌딩 블록"""
from __future__ import annotations
import torch, torch.nn as nn
from ..utils.config import CFG


def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, pad, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class ChannelAttn(nn.Module):
    def __init__(self, ch, r=None):
        super().__init__()
        mid = max(ch // (r or CFG.se_reduction), 4)
        self.mlp = nn.Sequential(nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, ch))
    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean(-1)) + self.mlp(x.max(-1).values))
        return x * w.unsqueeze(-1)


class TemporalAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        pool = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        return x * torch.sigmoid(self.conv(pool))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ChannelAttn(ch); self.t = TemporalAttn()
    def forward(self, x): return self.t(self.ch(x)) + x


class SEBlock(nn.Module):
    def __init__(self, ch, r=None):
        super().__init__()
        mid = max(ch // (r or CFG.se_reduction), 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.se(x).unsqueeze(-1)


class CrossGroupAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, CFG.cross_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.drop(out))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, k=7, attn="cbam"):
        super().__init__()
        pad = k // 2
        self.c1   = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
        self.b1   = nn.BatchNorm1d(out_ch)
        self.c2   = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.b2   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.attn_m = CBAM(out_ch) if attn=="cbam" else SEBlock(out_ch) if attn=="se" else nn.Identity()
    def forward(self, x):
        out = self.act(self.b1(self.c1(x)))
        return self.act(self.attn_m(self.b2(self.c2(out))) + self.skip(x))


class DilatedTCN(nn.Module):
    def __init__(self, ch, dilation, drop=0.3):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.net(x) + x)


class AttnPool1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        return (x * torch.softmax(self.score(x), dim=-1)).sum(-1)
