# -*- coding: utf-8 -*-
"""src/train/trainer.py — 핵심 학습 루프"""
from __future__ import annotations
import copy, gc
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from .losses   import build_criterion
from ..models.m1_flatcnn import FlatCNN
from ..utils.config import CFG, DEVICE, USE_AMP, AMP_DTYPE
from ..utils.logger import log


def _forward(model: nn.Module, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """배치 타입 자동 처리 → (logits_or_dict, y)"""
    is_hybrid = getattr(model, "IS_HYBRID", False)
    is_flat   = isinstance(model, FlatCNN)

    if is_hybrid and len(batch) == 3:
        bi, feat, yb = batch
        bi   = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
        feat = feat.to(DEVICE, non_blocking=True)
        out  = model(bi, feat)
    elif len(batch) == 2 and isinstance(batch[0], dict):
        bi, yb = batch
        bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
        if is_flat:
            # FlatCNN: dict → (B, C, T) concat
            x = torch.cat(list(bi.values()), dim=1)
            out = model(x)
        else:
            out = model(bi)
    else:
        xb, yb = batch
        out = model(xb.to(DEVICE, non_blocking=True))

    return out, yb


def _get_logits(out) -> torch.Tensor:
    if isinstance(out, dict):
        return out["final_logits"]
    return out


def _mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return x * lam + x[idx] * (1 - lam), y, y[idx], lam


class Trainer:
    def __init__(self, model: nn.Module, model_name: str, n_classes: int):
        self.model      = model.to(DEVICE)
        self.model_name = model_name
        self.criterion  = build_criterion(model_name, n_classes)
        self.opt        = torch.optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        self.scaler     = torch.amp.GradScaler("cuda", enabled=USE_AMP)
        self.best_f1    = -1.0
        self.best_state = None
        self.patience   = 0

    def _schedule(self, epochs: int):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=epochs, eta_min=CFG.min_lr
        )

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        for batch in loader:
            self.opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                out, yb = _forward(self.model, batch)
                logits  = _get_logits(out)
                yb      = yb.to(DEVICE)
                loss    = self.criterion(out, yb) if isinstance(out, dict) \
                          else self.criterion(logits, yb)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[list, list]:
        self.model.eval()
        yt, yp = [], []
        for batch in loader:
            out, yb = _forward(self.model, batch)
            pred    = _get_logits(out).argmax(1).cpu().numpy()
            yp.extend(pred.tolist())
            yt.extend(yb.cpu().numpy().tolist())
        return yt, yp

    def fit(
        self,
        tr_loader: DataLoader,
        te_loader: DataLoader,
        epochs: int    = None,
        early_stop: int = None,
        tag: str       = "",
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        epochs     = epochs     or CFG.epochs
        early_stop = early_stop or CFG.early_stop
        sch        = self._schedule(epochs)

        for ep in range(1, epochs + 1):
            self.train_epoch(tr_loader)
            sch.step()

            yt, yp = self.evaluate(te_loader)
            f1 = f1_score(yt, yp, average="macro", zero_division=0)

            if f1 > self.best_f1:
                self.best_f1    = f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience   = 0
            else:
                self.patience  += 1
                if self.patience >= early_stop:
                    break

            if ep % 10 == 0 or ep == epochs:
                acc = accuracy_score(yt, yp)
                log(f"  {tag} ep{ep:03d}  acc={acc:.4f}  f1={f1:.4f}  best={self.best_f1:.4f}")

        # 최고 모델 복원
        if self.best_state:
            self.model.load_state_dict(self.best_state)

        yt, yp  = self.evaluate(te_loader)
        acc     = accuracy_score(yt, yp)
        f1      = f1_score(yt, yp, average="macro", zero_division=0)
        return acc, f1, np.array(yt), np.array(yp)


def run_fold(
    model: nn.Module,
    model_name: str,
    n_classes: int,
    tr_loader: DataLoader,
    te_loader: DataLoader,
    tag: str = "",
    epochs: int    = None,
    early_stop: int = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """단일 fold 학습 → (acc, f1, y_true, y_pred)"""
    trainer = Trainer(model, model_name, n_classes)
    acc, f1, yt, yp = trainer.fit(tr_loader, te_loader, epochs, early_stop, tag)

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc, f1, yt, yp
