# -*- coding: utf-8 -*-
"""src/train/losses.py — 손실 함수"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from ..utils.config import CFG


class FocalLoss(nn.Module):
    """Focal Loss + Label Smoothing"""
    def __init__(self, gamma: float = None, smooth: float = None,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma  = gamma  or CFG.focal_gamma
        self.smooth = smooth or CFG.label_smooth
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)
        with torch.no_grad():
            td = torch.zeros_like(logits).fill_(self.smooth / max(n_cls - 1, 1))
            td.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)
        logp  = F.log_softmax(logits, dim=1)
        p     = torch.exp(logp)
        focal = (1.0 - p) ** self.gamma
        loss  = -(td * focal * logp).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class LabelSmoothCE(nn.Module):
    def __init__(self, smooth: float = None):
        super().__init__()
        self.smooth = smooth or CFG.label_smooth
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=self.smooth)


class FusionLoss(nn.Module):
    """FusionNet 전용 — 메인 loss + 속성 보조 loss"""
    def __init__(self, n_classes: int, aux_w: float = 0.3):
        super().__init__()
        self.main   = FocalLoss()
        self.bce    = nn.BCEWithLogitsLoss()
        self.ce3    = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.aux_w  = aux_w
        self.n_cls  = n_classes

    def forward(self, out: dict, y: torch.Tensor) -> torch.Tensor:
        main_loss = self.main(out["final_logits"], y)

        aux_loss = torch.tensor(0.0, device=y.device)
        if "slip_logit" in out:
            slip  = (y == 0).float()
            aux_loss = aux_loss + self.bce(out["slip_logit"], slip)
        if "irreg_logit" in out:
            irreg = ((y == 3) | (y == 4)).float()
            aux_loss = aux_loss + self.bce(out["irreg_logit"], irreg)
        if "flat_logit" in out:
            flat  = (y == 5).float()
            aux_loss = aux_loss + self.bce(out["flat_logit"], flat)
        if "slope_logits" in out:
            slope = torch.where(y == 1, torch.zeros_like(y),
                    torch.where(y == 2, torch.ones_like(y), torch.full_like(y, 2)))
            aux_loss = aux_loss + self.ce3(out["slope_logits"], slope)

        return main_loss + self.aux_w * aux_loss.detach()


def build_criterion(model_name: str, n_classes: int) -> nn.Module:
    if model_name == "FusionNet":
        return FusionLoss(n_classes)
    if CFG.use_focal:
        return FocalLoss()
    return LabelSmoothCE()
