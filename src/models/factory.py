from __future__ import annotations
import torch
import torch.nn as nn
from .m1_flatcnn    import FlatCNN
from .m2_branchcnn  import BranchCNN
from .m3_resnet1d   import ResNet1D
from .m4_resnet_tcn import ResNetTCN
from .m5_fusionnet  import FusionNet

MODEL_REGISTRY = {
    "FlatCNN":   FlatCNN,
    "BranchCNN": BranchCNN,
    "ResNet1D":  ResNet1D,
    "ResNetTCN": ResNetTCN,
    "FusionNet": FusionNet,
}
ALL_MODELS     = list(MODEL_REGISTRY.keys())
BASELINE_MODEL = "FlatCNN"
PROPOSED_MODEL = "FusionNet"


class _FeatureMLP(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=512, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(drop*0.5),
            nn.Linear(hidden//2, n_cls),
        )
    def forward(self, x): return self.net(x)


def build_model(name, branch_ch, num_classes, n_feat=None, input_mode="raw"):
    if input_mode == "feat":
        assert n_feat is not None
        return _FeatureMLP(n_feat, num_classes)

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {ALL_MODELS}")

    cls = MODEL_REGISTRY[name]

    if name == "FlatCNN":
        total_ch = sum(branch_ch.values())
        return cls(in_ch=total_ch, num_classes=num_classes)
    if name == "FusionNet":
        return cls(branch_ch=branch_ch, num_classes=num_classes, n_feat=n_feat)
    return cls(branch_ch=branch_ch, num_classes=num_classes)
