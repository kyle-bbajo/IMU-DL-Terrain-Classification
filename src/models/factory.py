# -*- coding: utf-8 -*-
"""src/models/factory.py — 모델 팩토리"""
from __future__ import annotations
from .m1_flatcnn   import FlatCNN
from .m2_branchcnn import BranchCNN
from .m3_resnet1d  import ResNet1D
from .m4_resnet_tcn import ResNetTCN
from .m5_fusionnet  import FusionNet

# 논문 이름 → 클래스 매핑
MODEL_REGISTRY: dict[str, type] = {
    "FlatCNN":   FlatCNN,
    "BranchCNN": BranchCNN,
    "ResNet1D":  ResNet1D,
    "ResNetTCN": ResNetTCN,
    "FusionNet": FusionNet,
}

ALL_MODELS     = list(MODEL_REGISTRY.keys())
BASELINE_MODEL = "FlatCNN"
PROPOSED_MODEL = "FusionNet"


def build_model(
    name: str,
    branch_ch: dict[str, int],
    num_classes: int,
    n_feat: int | None = None,
) -> object:
    """
    모델 이름 + 설정으로 인스턴스 생성.

    Parameters
    ----------
    name       : MODEL_REGISTRY 키
    branch_ch  : 센서 그룹별 채널 수
    num_classes: 분류 클래스 수
    n_feat     : 도메인 피처 차원 (FusionNet만 필요)
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {ALL_MODELS}")

    cls = MODEL_REGISTRY[name]

    if name == "FlatCNN":
        # 54ch 전체를 flat input으로
        total_ch = sum(branch_ch.values())
        return cls(in_ch=total_ch, num_classes=num_classes)

    if name == "FusionNet":
        return cls(branch_ch=branch_ch, num_classes=num_classes, n_feat=n_feat)

    return cls(branch_ch=branch_ch, num_classes=num_classes)
