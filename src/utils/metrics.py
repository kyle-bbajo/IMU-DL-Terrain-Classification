# -*- coding: utf-8 -*-
"""src/utils/metrics.py — 평가 지표"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, classification_report,
)
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc":              round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1":         round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class_recall": [round(float(r), 4) for r in
                             recall_score(y_true, y_pred, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_cm(y_true, y_pred, class_names, tag, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{tag}.png", dpi=150)
    plt.close()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


class Timer:
    def __init__(self):
        import time
        self._s = time.time()
    def elapsed(self) -> float:
        import time
        return time.time() - self._s
    def __str__(self):
        e = self.elapsed()
        m, s = divmod(int(e), 60)
        return f"{m}m{s:02d}s" if m else f"{s}s"
    def __enter__(self): return self
    def __exit__(self, *_): pass
