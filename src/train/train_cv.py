# -*- coding: utf-8 -*-
"""src/train/train_cv.py — 5-fold CV / LOSO 공통 실험 루프"""
from __future__ import annotations
import gc, json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .trainer          import run_fold
from ..data.loader     import build_loaders
from ..data.split      import kfold_splits, loso_splits
from ..models.factory  import build_model
from ..features.builder import get_feat_mask, combo_name
from ..utils.config    import CFG, CLASS_NAMES
from ..utils.logger    import log
from ..utils.metrics   import compute_metrics, save_json, Timer
from ..utils.db import save_result


def run_kfold(
    exp_name:   str,
    model_name: str,
    input_mode: str,
    feat_combo: list[str],
    X_all:      np.ndarray,
    feat_all:   np.ndarray,
    y_all:      np.ndarray,
    groups:     np.ndarray,
    branch_idx: dict,
    branch_ch:  dict,
    n_classes:  int,
    out_dir:    Path,
    epochs:     int   = None,
    early_stop: int   = None,
    kfold:      int   = None,
    seed:       int   = 42,
) -> dict:
    """5-fold CV 실험 → 결과 dict 저장"""
    res_path = out_dir / f"{exp_name}.json"
    if res_path.exists():
        log(f"  ⏭ skip: {exp_name}")
        return json.loads(res_path.read_text())

    log(f"\n  ── {exp_name}")
    kfold = kfold or CFG.kfold

    # 피처 마스킹
    feat_use, feat_dim = _apply_feat_mask(feat_all, feat_combo, input_mode)

    splits = kfold_splits(y_all, groups, n_splits=kfold, seed=seed)
    all_yt, all_yp, fold_res = [], [], []

    with Timer() as t:
        for fi, (tr_idx, te_idx) in enumerate(splits, 1):
            ft_tr, ft_te = _normalize(feat_use, tr_idx, te_idx, feat_dim)
            tr_dl, te_dl = build_loaders(
                X_all[tr_idx], ft_tr, y_all[tr_idx],
                X_all[te_idx], ft_te, y_all[te_idx],
                branch_idx, input_mode,
            )
            model = build_model(model_name, branch_ch, n_classes, n_feat=feat_dim, input_mode=input_mode)
            acc, f1, yt, yp = run_fold(
                model, model_name, n_classes, tr_dl, te_dl,
                tag=f"[{model_name}|{input_mode}|F{fi}]",
                epochs=epochs, early_stop=early_stop,
            )
            all_yt.extend(yt.tolist()); all_yp.extend(yp.tolist())
            fold_res.append({"fold": fi, "acc": round(acc, 4), "f1": round(f1, 4)})
            del model, tr_dl, te_dl; gc.collect()

    result = _aggregate(exp_name, model_name, input_mode, feat_combo, feat_dim,
                        all_yt, all_yp, n_classes, fold_res, str(t))
    save_json(result, res_path)
    save_result(result)
    log(f"  ★ {exp_name}  Acc={result['acc']:.4f}  F1={result['macro_f1']:.4f}  ({t})")
    return result


def run_loso(
    exp_name:   str,
    model_name: str,
    input_mode: str,
    feat_combo: list[str],
    X_all:      np.ndarray,
    feat_all:   np.ndarray,
    y_all:      np.ndarray,
    groups:     np.ndarray,
    branch_idx: dict,
    branch_ch:  dict,
    n_classes:  int,
    out_dir:    Path,
    epochs:     int = None,
    early_stop: int = None,
) -> dict:
    """LOSO 실험 → 결과 dict 저장"""
    res_path = out_dir / f"loso_{exp_name}.json"
    if res_path.exists():
        log(f"  ⏭ skip (LOSO): {exp_name}")
        return json.loads(res_path.read_text())

    log(f"\n  ── LOSO: {exp_name}")
    feat_use, feat_dim = _apply_feat_mask(feat_all, feat_combo, input_mode)

    splits   = loso_splits(groups)
    all_yt, all_yp, per_subj = [], [], {}

    with Timer() as t:
        for si, (sid, tr_idx, te_idx) in enumerate(splits, 1):
            ft_tr, ft_te = _normalize(feat_use, tr_idx, te_idx, feat_dim)
            tr_dl, te_dl = build_loaders(
                X_all[tr_idx], ft_tr, y_all[tr_idx],
                X_all[te_idx], ft_te, y_all[te_idx],
                branch_idx, input_mode,
            )
            model = build_model(model_name, branch_ch, n_classes, n_feat=feat_dim, input_mode=input_mode)
            acc, f1, yt, yp = run_fold(
                model, model_name, n_classes, tr_dl, te_dl,
                tag=f"[LOSO|S{sid:02d}|{model_name}]",
                epochs=epochs, early_stop=early_stop,
            )
            all_yt.extend(yt.tolist()); all_yp.extend(yp.tolist())
            per_subj[int(sid)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
            log(f"  [{si:02d}/{len(splits)}] S{sid:02d}  acc={acc:.4f}  f1={f1:.4f}")
            del model, tr_dl, te_dl; gc.collect()

    result = _aggregate(exp_name, model_name, input_mode, feat_combo, feat_dim,
                        all_yt, all_yp, n_classes, [], str(t))
    result["per_subject"] = per_subj
    save_json(result, res_path)
    save_result(result)
    log(f"  ★ LOSO {exp_name}  Acc={result['acc']:.4f}  F1={result['macro_f1']:.4f}  ({t})")
    return result


# ─────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────

def _apply_feat_mask(feat_all, feat_combo, input_mode):
    if feat_combo and input_mode != "raw":
        fidx     = get_feat_mask(feat_combo, n_features=feat_all.shape[1])
        feat_use = feat_all[:, fidx]
    else:
        feat_use = feat_all
    feat_dim = feat_use.shape[1] if input_mode != "raw" else 0
    return feat_use, feat_dim


def _normalize(feat_use, tr_idx, te_idx, feat_dim):
    if feat_dim > 0:
        sc    = StandardScaler()
        ft_tr = sc.fit_transform(feat_use[tr_idx])
        ft_te = sc.transform(feat_use[te_idx])
    else:
        ft_tr = feat_use[tr_idx]
        ft_te = feat_use[te_idx]
    return ft_tr, ft_te


def _aggregate(exp_name, model_name, input_mode, feat_combo, feat_dim,
               all_yt, all_yp, n_classes, fold_res, elapsed):
    yt_a    = np.array(all_yt); yp_a = np.array(all_yp)
    acc     = round(accuracy_score(yt_a, yp_a), 4)
    f1      = round(f1_score(yt_a, yp_a, average="macro", zero_division=0), 4)
    recalls = recall_score(yt_a, yp_a, average=None, zero_division=0)
    return {
        "exp_name":   exp_name,
        "model":      model_name,
        "input_mode": input_mode,
        "feat_combo": feat_combo,
        "feat_dim":   int(feat_dim),
        "acc":        acc,
        "macro_f1":   f1,
        "per_class_recall": {
            CLASS_NAMES[i]: round(float(r), 4)
            for i, r in enumerate(recalls) if i < len(CLASS_NAMES)
        },
        "folds":   fold_res,
        "elapsed": elapsed,
    }
