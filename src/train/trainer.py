from __future__ import annotations
import copy, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from .losses   import build_criterion
from ..utils.config import CFG, DEVICE, USE_AMP, AMP_DTYPE
from ..utils.logger import log


def _forward(model: nn.Module, batch):
    is_hybrid = getattr(model, "IS_HYBRID", False)
    is_flat   = model.__class__.__name__ in ("FlatCNN", "_FeatureMLP")

    if len(batch) == 3:
        bi, feat, yb = batch
        if isinstance(bi, dict):
            bi   = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            feat = feat.to(DEVICE, non_blocking=True)
            out  = model(bi, feat) if is_hybrid else model(bi)
        else:
            out = model(bi.to(DEVICE, non_blocking=True))
    elif len(batch) == 2 and isinstance(batch[0], dict):
        bi, yb = batch
        bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
        if is_flat:
            out = model(torch.cat(list(bi.values()), dim=1))
        else:
            out = model(bi)
    else:
        xb, yb = batch
        out = model(xb.to(DEVICE, non_blocking=True))

    if isinstance(out, dict):
        out = out["final_logits"]
    return out, yb


class Trainer:
    def __init__(self, model, model_name, n_classes):
        self.model      = model.to(DEVICE)
        self.model_name = model_name
        self.criterion  = build_criterion(model_name, n_classes)
        self.opt        = torch.optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        self.scaler     = torch.amp.GradScaler("cuda", enabled=USE_AMP)
        self.best_f1    = -1.0
        self.best_state = None
        self.patience   = 0

    def _schedule(self, epochs):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=epochs, eta_min=CFG.min_lr)

    def train_epoch(self, loader):
        self.model.train()
        for batch in loader:
            self.opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                out, yb = _forward(self.model, batch)
                loss    = self.criterion(out, yb.to(DEVICE))
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        yt, yp = [], []
        for batch in loader:
            out, yb = _forward(self.model, batch)
            yp.extend(out.argmax(1).cpu().numpy().tolist())
            yt.extend(yb.cpu().numpy().tolist())
        return yt, yp

    def fit(self, tr_loader, te_loader, epochs=None, early_stop=None, tag=""):
        epochs     = epochs     or CFG.epochs
        early_stop = early_stop or CFG.early_stop
        sch        = self._schedule(epochs)

        for ep in range(1, epochs + 1):
            self.train_epoch(tr_loader)
            sch.step()
            yt, yp = self.evaluate(te_loader)
            f1     = f1_score(yt, yp, average="macro", zero_division=0)

            if f1 > self.best_f1:
                self.best_f1, self.best_state, self.patience = \
                    f1, copy.deepcopy(self.model.state_dict()), 0
            else:
                self.patience += 1
                if self.patience >= early_stop: break

            if ep % 10 == 0 or ep == epochs:
                log(f"  {tag} ep{ep:03d}  acc={accuracy_score(yt,yp):.4f}  f1={f1:.4f}")

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        yt, yp = self.evaluate(te_loader)
        return accuracy_score(yt,yp), f1_score(yt,yp,average="macro",zero_division=0), \
               np.array(yt), np.array(yp)


def run_fold(model, model_name, n_classes, tr_loader, te_loader,
             tag="", epochs=None, early_stop=None):
    trainer = Trainer(model, model_name, n_classes)
    result  = trainer.fit(tr_loader, te_loader, epochs, early_stop, tag)
    del trainer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result
