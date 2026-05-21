"""ResNet backbone + MLP head for Geometry Dash difficulty prediction."""

import torch
import torch.nn as nn
import torchvision.models as tvm

# Feature dimension after global average pooling for each ResNet variant
RESNET_OUT_DIMS: dict[str, int] = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


class GeometryDashCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        resnet_version: str = "resnet50",
        hidden_dims: tuple[int, ...] = (512, 256),
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of difficulty classes (10 for 1-star to 10-star).
            resnet_version: One of resnet18 / resnet34 / resnet50 / resnet101 / resnet152.
            hidden_dims: Sizes of hidden FC layers before the final classifier.
            pretrained: Load ImageNet-pretrained weights.
            dropout: Dropout probability between FC layers.
        """
        super().__init__()

        if resnet_version not in RESNET_OUT_DIMS:
            raise ValueError(f"resnet_version must be one of {list(RESNET_OUT_DIMS)}")

        weights_enum = tvm.get_model_weights(resnet_version)
        weights = weights_enum.DEFAULT if pretrained else None
        backbone = tvm.get_model(resnet_version, weights=weights)
        feature_dim = RESNET_OUT_DIMS[resnet_version]

        # Drop the final FC layer; keep everything up to (and including) avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Build MLP head
        layers: list[nn.Module] = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, frames_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            frames_list: List of (T_i, C, H, W) tensors — one per sample in the batch.
                         T_i may differ across samples.
        Returns:
            Logits of shape (B, num_classes).
        """
        pooled = []
        for frames in frames_list:
            # frames: (T, C, H, W)
            features = self.backbone(frames)          # (T, feature_dim, 1, 1)
            features = features.flatten(1)            # (T, feature_dim)
            pooled.append(features.mean(dim=0))       # (feature_dim,)

        x = torch.stack(pooled)  # (B, feature_dim)
        return self.head(x)      # (B, num_classes)
