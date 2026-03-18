# -*- coding: utf-8 -*-
"""
src/utils/config.py — 전역 설정
IMU-DL-Terrain-Classification
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time, os, torch

# ─────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # repo root
DATA_DIR   = ROOT / "data"
RESULT_DIR = ROOT / "results"
LOG_DIR    = ROOT / "logs"
CACHE_DIR  = ROOT / "data" / "cache"

H5_PATH    = DATA_DIR / "dataset.h5"   # symlink or 실제 파일

for _d in [DATA_DIR, RESULT_DIR, LOG_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def make_result_dir(exp_name: str) -> Path:
    ts  = time.strftime("%Y%m%d_%H%M%S")
    out = RESULT_DIR / f"{ts}_{exp_name}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out


# ─────────────────────────────────────────────
# GPU
# ─────────────────────────────────────────────
USE_GPU   = torch.cuda.is_available()
DEVICE    = torch.device("cuda" if USE_GPU else "cpu")
USE_AMP   = USE_GPU
AMP_DTYPE = torch.bfloat16  # L4: BF16 지원


# ─────────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────────
@dataclass
class Config:
    # 데이터
    n_subjects:   int   = 50
    num_classes:  int   = 6
    sample_rate:  int   = 200
    ts:           int   = 256       # 스텝 길이 (포인트)

    # 학습
    kfold:        int   = 5
    epochs:       int   = 80
    early_stop:   int   = 15
    batch:        int   = 2048      # L4 24GB → 큰 배치
    lr:           float = 1e-3
    min_lr:       float = 1e-6
    weight_decay: float = 1e-3
    dropout_clf:  float = 0.5
    dropout_feat: float = 0.3
    label_smooth: float = 0.1
    mixup_alpha:  float = 0.2

    # 기법
    use_focal:    bool  = True
    focal_gamma:  float = 2.0
    use_balanced: bool  = True
    use_fft:      bool  = True
    fft_group:    str   = "Foot"
    grad_clip:    float = 1.0

    # 모델 구조
    feat_dim:     int   = 256
    se_reduction: int   = 4
    cross_heads:  int   = 4

    # 증강
    aug_noise:    float = 0.03
    aug_scale:    float = 0.15
    aug_shift:    int   = 15
    aug_mask:     float = 0.05

    # 피처
    n_features:   int   = 390

    # g6.4xlarge 최적화 (vCPU 16, RAM 64GB)
    num_workers:  int   = 8
    pin_memory:   bool  = True


CFG = Config()

# ─────────────────────────────────────────────
# 클래스 정보
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "C1-미끄러운", "C2-오르막", "C3-내리막",
    "C4-흙길",     "C5-잔디",   "C6-평지",
]

CLASS_EN = [
    "Slippery", "Uphill", "Downhill",
    "Dirt",     "Grass",  "Flat",
]
