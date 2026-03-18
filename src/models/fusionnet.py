# -*- coding: utf-8 -*-
"""
src/models/fusionnet.py — FusionNet (제안 모델)
브랜치 CNN + 도메인 피처 MLP 융합 + 속성 보조 헤드
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from .blocks import ConvBNReLU, CBAM, CrossGroupAttn, _init
from ..utils.config import CFG, N_FEATURES  # type: ignore


def _augment(x, training):
    if not training: return x
    b, c, t = x.shape
    if CFG.aug_noise > 0: x = x + torch.randn_like(x) * CFG.aug_noise
    if CFG.aug_scale > 0: x = x * (1.0 + (torch.rand(b,1,1,device=x.device)-0.5)*CFG.aug_scale)
    return x


class _Branch(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        d = CFG.dropout_feat
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(64, 128, 5, 2),   nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(128, out_dim, 3, 1),
        )
        self.attn = CBAM(out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x): return self.pool(self.attn(self.net(x))).squeeze(-1)


class _FreqBranch(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2),
            ConvBNReLU(64, out_dim, 5, 2), nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x): return self.net(torch.fft.rfft(x, dim=-1).abs()).squeeze(-1)


class _FeatureMLP(nn.Module):
    def __init__(self, n_feat, out_dim):
        super().__init__()
        d = CFG.dropout_feat
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Linear(n_feat, 128), nn.ReLU(inplace=True), nn.Dropout(d),
            nn.Linear(128, out_dim), nn.ReLU(inplace=True), nn.Dropout(d),
        )
    def forward(self, f): return self.net(f.float())


def _build_attr_targets(y: torch.Tensor):
    """y (0~5) → 속성 보조 레이블"""
    # slip: C1(미끄러운)=1
    slip  = (y == 0).float()
    # slope: C2(오르막)=0, C3(내리막)=1, 나머지=2
    slope = torch.where(y == 1, torch.zeros_like(y),
            torch.where(y == 2, torch.ones_like(y), torch.full_like(y, 2)))
    # irreg: C4(흙길)/C5(잔디)=1
    irreg = ((y == 3) | (y == 4)).float()
    # flat: C6(평지)=1
    flat  = (y == 5).float()
    return slip, slope, irreg, flat


class FusionNet(nn.Module):
    """
    제안 모델 (FusionNet).
    브랜치 CNN + 도메인 피처(390차원) MLP 융합 + 속성 보조 헤드.
    IS_HYBRID = True → HierarchicalDataset 사용.
    """
    IS_HYBRID = True

    def __init__(self, branch_ch: dict[str, int], num_classes: int = CFG.num_classes,
                 n_feat: int = None):
        super().__init__()
        n_feat = n_feat or getattr(CFG, "n_features", 390)
        feat   = CFG.feat_dim; dc = CFG.dropout_clf

        self.names    = list(branch_ch.keys()); self.n = len(self.names)
        self.branches = nn.ModuleDict({nm: _Branch(ch, feat) for nm, ch in branch_ch.items()})
        self.cross    = CrossGroupAttn(feat)

        self.use_fft = CFG.use_fft and CFG.fft_group in branch_ch
        if self.use_fft:
            self.fft_src     = CFG.fft_group
            self.freq_branch = _FreqBranch(branch_ch[self.fft_src], feat)

        n_f        = self.n + (1 if self.use_fft else 0)
        cnn_dim    = feat * n_f
        self.feat_mlp  = _FeatureMLP(n_feat, feat)
        fusion_dim     = cnn_dim + feat
        self.bn_fuse   = nn.BatchNorm1d(fusion_dim)

        # 속성 보조 헤드
        self.slip_head  = nn.Linear(fusion_dim, 1)
        self.slope_head = nn.Linear(fusion_dim, 3)
        self.irreg_head = nn.Linear(fusion_dim, 1)
        self.flat_head  = nn.Linear(fusion_dim, 1)

        # 메인 분류기: fusion + aux(6차원) 입력
        self.clf = nn.Sequential(
            nn.Linear(fusion_dim + 6, 256), nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(256, 128),            nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(128, num_classes),
        )
        self.apply(_init)

    def _fuse(self, bi, feat):
        feats = [self.branches[nm](bi[nm]) for nm in self.names]
        x     = self.cross(torch.stack(feats, dim=1))
        feats = [x[:, i, :] for i in range(self.n)]
        if self.use_fft: feats.append(self.freq_branch(bi[self.fft_src]))
        return self.bn_fuse(torch.cat([torch.cat(feats, dim=1), self.feat_mlp(feat)], dim=1))

    def forward(self, bi, feat):
        if self.training: bi = {k: _augment(v, True) for k, v in bi.items()}
        h   = self._fuse(bi, feat)

        # 속성 헤드 (있을 때만)
        sl  = self.slip_head(h)  if hasattr(self, "slip_head")  else torch.zeros(h.size(0),1,device=h.device)
        slo = self.slope_head(h) if hasattr(self, "slope_head") else torch.zeros(h.size(0),3,device=h.device)
        ir  = self.irreg_head(h) if hasattr(self, "irreg_head") else torch.zeros(h.size(0),1,device=h.device)
        fl  = self.flat_head(h)  if hasattr(self, "flat_head")  else torch.zeros(h.size(0),1,device=h.device)

        aux = torch.cat([sl, slo, ir, fl], dim=-1)
        fin = self.clf(torch.cat([h, aux], dim=-1))

        return {
            "final_logits": fin,
            "slip_logit":   sl.squeeze(-1),
            "slope_logits": slo,
            "irreg_logit":  ir.squeeze(-1),
            "flat_logit":   fl.squeeze(-1),
        }

    def extract(self, bi, feat): return self._fuse(bi, feat)
