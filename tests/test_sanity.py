# -*- coding: utf-8 -*-
"""
tests/test_sanity.py — 단위 테스트 (sanity check)

각 모듈의 import, shape, forward pass 를 빠르게 확인.
전체 학습 없이 코드 구조가 올바른지만 검증.

실행:
  python tests/test_sanity.py
  python -m pytest tests/test_sanity.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PILOT = Path("/home/ubuntu/project/repo/src")
sys.path.insert(0, str(PILOT))

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 4      # 배치 크기
C = 54     # 채널 수
T = 256    # 타임스텝
N_FEAT = 390


def _dummy_branch_ch():
    return {"Pelvis": 6, "Hand": 12, "Thigh": 12, "Shank": 12, "Foot": 12}


def _dummy_bi(branch_ch):
    return {nm: torch.randn(B, ch, T).to(DEVICE) for nm, ch in branch_ch.items()}


def _dummy_feat():
    return torch.randn(B, N_FEAT).to(DEVICE)


# ─────────────────────────────────────────────
# 1. utils
# ─────────────────────────────────────────────
def test_config():
    from src.utils.config import CFG, CLASS_NAMES
    assert CFG.num_classes == 6
    assert len(CLASS_NAMES) == 6
    print("  ✅ config")


def test_seed():
    from src.utils.seed import seed_everything
    seed_everything(42)
    print("  ✅ seed")


def test_metrics():
    from src.utils.metrics import compute_metrics, Timer
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 5])
    m = compute_metrics(y_true, y_pred)
    assert m["acc"] == 1.0
    print("  ✅ metrics")


# ─────────────────────────────────────────────
# 2. features
# ─────────────────────────────────────────────
def test_features_builder():
    from src.features.builder import (
        get_feat_mask, get_sensor_mask,
        all_feat_type_combos, sensor_cumulative_combos, combo_name,
    )
    mask = get_feat_mask(["TIME", "FREQ"], n_features=390)
    assert len(mask) > 0 and mask.max() < 390
    mask2 = get_sensor_mask(["Foot", "Shank"], n_features=390)
    assert len(mask2) > 0
    combos = all_feat_type_combos(include_context=False)
    assert len(combos) == 15   # 2^4 - 1
    combos_ctx = all_feat_type_combos(include_context=True)
    assert len(combos_ctx) == 31  # 2^5 - 1
    sensor_combos = sensor_cumulative_combos()
    assert len(sensor_combos) == 6
    assert combo_name(["TIME", "FREQ"]) == "T+F"
    print(f"  ✅ features/builder  feat_mask={len(mask)}  combos={len(combos)}")


# ─────────────────────────────────────────────
# 3. data
# ─────────────────────────────────────────────
def test_datasets():
    from src.data.dataset import BranchDataset, FlatDataset, FeatDataset, FusionDataset
    branch_ch = _dummy_branch_ch()
    N = 20
    X     = np.random.randn(N, T, C).astype(np.float32)
    feat  = np.random.randn(N, N_FEAT).astype(np.float32)
    y     = np.random.randint(0, 6, N).astype(np.int64)

    # BranchDataset
    from channel_groups import build_branch_idx
    channels = [f"ch{i}" for i in range(C)]
    # 간단히 branch_idx 수동 생성
    branch_idx = {
        "Pelvis": list(range(0,6)),   "Hand":  list(range(6,18)),
        "Thigh":  list(range(18,30)), "Shank": list(range(30,42)),
        "Foot":   list(range(42,54)),
    }
    ds = BranchDataset(X, y, branch_idx)
    bi, lbl = ds[0]
    assert isinstance(bi, dict) and lbl.dtype == torch.int64

    ds2 = FlatDataset(X, y)
    x2, lbl2 = ds2[0]
    assert x2.shape == (C, T)

    ds3 = FeatDataset(feat, y)
    x3, lbl3 = ds3[0]
    assert x3.shape == (N_FEAT,)

    ds4 = FusionDataset(X, feat, y, branch_idx)
    bi4, f4, lbl4 = ds4[0]
    assert f4.shape == (N_FEAT,)
    print("  ✅ data/dataset")


def test_split():
    from src.data.split import kfold_splits, loso_splits
    N = 50
    y      = np.random.randint(0, 6, N)
    groups = np.repeat(np.arange(10), 5)
    splits = kfold_splits(y, groups, n_splits=5)
    assert len(splits) == 5
    loso = loso_splits(groups)
    assert len(loso) == 10
    print("  ✅ data/split")


# ─────────────────────────────────────────────
# 4. models
# ─────────────────────────────────────────────
def test_flatcnn():
    from src.models.flatcnn import FlatCNN
    model = FlatCNN(in_ch=C, num_classes=6).to(DEVICE)
    x = torch.randn(B, C, T).to(DEVICE)
    out = model(x)
    assert out.shape == (B, 6)
    print(f"  ✅ FlatCNN  params={sum(p.numel() for p in model.parameters()):,}")


def test_branchcnn():
    from src.models.branchcnn import BranchCNN
    branch_ch = _dummy_branch_ch()
    model = BranchCNN(branch_ch, num_classes=6).to(DEVICE)
    bi  = _dummy_bi(branch_ch)
    out = model(bi)
    assert out.shape == (B, 6)
    print(f"  ✅ BranchCNN  params={sum(p.numel() for p in model.parameters()):,}")


def test_resnet1d():
    from src.models.resnet1d import ResNet1D
    branch_ch = _dummy_branch_ch()
    model = ResNet1D(branch_ch, num_classes=6).to(DEVICE)
    bi  = _dummy_bi(branch_ch)
    out = model(bi)
    assert out.shape == (B, 6)
    print(f"  ✅ ResNet1D  params={sum(p.numel() for p in model.parameters()):,}")


def test_resnet_tcn():
    from src.models.resnet_tcn import ResNetTCN
    branch_ch = _dummy_branch_ch()
    model = ResNetTCN(branch_ch, num_classes=6).to(DEVICE)
    bi  = _dummy_bi(branch_ch)
    out = model(bi)
    assert out.shape == (B, 6)
    print(f"  ✅ ResNetTCN  params={sum(p.numel() for p in model.parameters()):,}")


def test_fusionnet():
    from src.models.fusionnet import FusionNet
    branch_ch = _dummy_branch_ch()
    model = FusionNet(branch_ch, num_classes=6, n_feat=N_FEAT).to(DEVICE)
    bi   = _dummy_bi(branch_ch)
    feat = _dummy_feat()
    out  = model(bi, feat)
    assert isinstance(out, dict)
    assert out["final_logits"].shape == (B, 6)
    print(f"  ✅ FusionNet  params={sum(p.numel() for p in model.parameters()):,}")


def test_factory():
    from src.models.factory import build_model, ALL_MODELS
    assert len(ALL_MODELS) == 5
    branch_ch = _dummy_branch_ch()
    for name in ALL_MODELS:
        m = build_model(name, branch_ch, n_classes=6, n_feat=N_FEAT)
        assert m is not None
    print(f"  ✅ factory  models={ALL_MODELS}")


# ─────────────────────────────────────────────
# 5. train
# ─────────────────────────────────────────────
def test_losses():
    from src.train.losses import FocalLoss, LabelSmoothCE, FusionLoss, build_criterion
    logits = torch.randn(B, 6)
    y      = torch.randint(0, 6, (B,))
    fl  = FocalLoss()(logits, y)
    ce  = LabelSmoothCE()(logits, y)
    assert fl.item() > 0 and ce.item() > 0

    # FusionNet loss
    out = {
        "final_logits": logits,
        "slip_logit":   torch.randn(B),
        "slope_logits": torch.randn(B, 3),
        "irreg_logit":  torch.randn(B),
        "flat_logit":   torch.randn(B),
    }
    fl2 = FusionLoss(6)(out, y)
    assert fl2.item() > 0
    print("  ✅ losses")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
TESTS = [
    ("utils/config",    test_config),
    ("utils/seed",      test_seed),
    ("utils/metrics",   test_metrics),
    ("features/builder",test_features_builder),
    ("data/dataset",    test_datasets),
    ("data/split",      test_split),
    ("models/FlatCNN",  test_flatcnn),
    ("models/BranchCNN",test_branchcnn),
    ("models/ResNet1D", test_resnet1d),
    ("models/ResNetTCN",test_resnet_tcn),
    ("models/FusionNet",test_fusionnet),
    ("models/factory",  test_factory),
    ("train/losses",    test_losses),
]


def main():
    print("=" * 60)
    print("  Sanity Check — IMU-DL-Terrain-Classification")
    print(f"  device={DEVICE}")
    print("=" * 60)

    passed, failed = [], []
    for name, fn in TESTS:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"  결과: {len(passed)}/{len(TESTS)} 통과")
    if failed:
        print(f"  실패: {failed}")
    else:
        print("  ✅ 모든 테스트 통과")
    print("=" * 60)
    return len(failed) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
