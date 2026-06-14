"""The two model pieces:

1. build_backbone() -> a FROZEN pretrained CNN that turns an image into a
   fixed-length feature vector. We never train it; that is what keeps this
   project runnable on a CPU.
2. ClassifierHead -> the small network we actually train, on top of the cached
   features. It outputs one raw logit per disease (multi-label).
"""
import torch.nn as nn
import torchvision

import config


def build_backbone(name: str = config.BACKBONE):
    """Return (backbone, feature_dim).

    The backbone ends with a global-average-pool + flatten, so its output is a
    single vector per image. All parameters are frozen and it is set to eval().
    """
    if name == "mobilenet_v2":
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        net = torchvision.models.mobilenet_v2(weights=weights)
        features = net.features
        feature_dim = 1280
    elif name == "densenet121":
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        net = torchvision.models.densenet121(weights=weights)
        features = net.features
        feature_dim = 1024
    else:
        raise ValueError(f"Unknown backbone: {name!r} (use 'mobilenet_v2' or 'densenet121')")

    backbone = nn.Sequential(
        features,
        nn.ReLU(inplace=True),         # densenet's features end in a BatchNorm; relu here. Harmless for mobilenet.
        nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, C, 1, 1)  global average pool
        nn.Flatten(),                  # -> (B, C)
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, feature_dim


class ClassifierHead(nn.Module):
    """Small trainable head. Input: feature vector. Output: raw logits (one per class).

    We use raw logits (no sigmoid here) because the loss is BCEWithLogitsLoss,
    which applies the sigmoid internally and is numerically more stable.
    """

    def __init__(self, in_dim: int,
                 num_classes: int = config.NUM_CLASSES,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
