import torch
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


class _FeatureMLP(torch.nn.Module):
    """피처만 입력하는 MLP (feat 모드용, 모든 모델 공통)"""
    def __init__(self, in_dim, n_cls, hidden=512, drop=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_dim),
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
            torch.nn.Linear(hidden, hidden//2), torch.nn.GELU(), torch.nn.Dropout(drop*0.5),
            torch.nn.Linear(hidden//2, n_cls),
        )
    def forward(self, x): return self.net(x)


def build_model(
    name: str,
    branch_ch: dict[str, int],
    num_classes: int,
    n_feat: int | None = None,
    input_mode: str = "raw",
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
    # feat 모드: 모든 모델이 FeatureMLP 사용
    if input_mode == "feat":
        assert n_feat is not None, "feat 모드에서 n_feat 필요"
        return _FeatureMLP(n_feat, num_classes)

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
