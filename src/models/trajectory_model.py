"""
Trajectory prediction model with SlowFast backbone.

Architecture:
- Input: 16 frames at 224×224 resolution (16, 3, 224, 224)
- SlowFast backbone: processes fast and slow pathways
- Aggregation: flattens backbone output to (B, D) vector
- Trajectory head: MLP predicting 32 future heights
"""

import torch
import torch.nn as nn
from .trajectory_head import TrajectoryHead


class TrajectoryModel(nn.Module):
    """
    Trajectory prediction model with SlowFast backbone.

    Maps 16-frame video input to 32-frame height predictions.
    """

    def __init__(self, backbone, head_input_dim: int = 2048, hidden_dim: int = 512):
        """
        Args:
            backbone: SlowFast model (or any backbone that outputs a tensor)
            head_input_dim: Output dimension from backbone (default 2048 for SlowFast)
            hidden_dim: Hidden dimension in trajectory head
        """
        super().__init__()
        self.backbone = backbone
        self.head = TrajectoryHead(head_input_dim, hidden_dim=hidden_dim)

    def forward(self, frames):
        """
        Args:
            frames: (batch_size, 16, 3, 224, 224) video frames

        Returns:
            heights: (batch_size, 32) predicted heights
        """
        # Extract backbone features
        # Input shape expected by SlowFast: (B, 3, T, H, W)
        # frames is currently (B, T, 3, H, W), so permute
        frames = frames.permute(0, 2, 1, 3, 4)  # (B, 3, 16, 224, 224)

        # Get backbone output
        backbone_out = self.backbone(frames)

        # Flatten backbone output to (B, D)
        if backbone_out.dim() > 1:
            backbone_features = backbone_out.view(backbone_out.size(0), -1)
        else:
            backbone_features = backbone_out

        # Predict heights via MLP head
        heights = self.head(backbone_features)  # (B, 32)

        return heights
