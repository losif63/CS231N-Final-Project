"""
Loss functions for trajectory prediction.

Three-component loss:
1. Smooth L1 loss for spatial accuracy (β=0.05)
2. L1 loss on velocity deltas for kinematic regularization (λ=0.2)
3. Per-frame occlusion weighting (1 - occlusion_i)

Written using Claude Code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLoss(nn.Module):
    """
    Combined loss for trajectory prediction with occlusion weighting.

    L = (1/32) * Σ w_i * [L_smooth(i) + λ * L_vel(i)]
    where:
      w_i = 1 - occlusion_i (per-frame weight)
      L_smooth = SmoothL1 with β=0.05
      L_vel = L1 loss on frame-to-frame velocity differences
      λ = 0.2 (velocity penalty weight)
    """

    def __init__(self, beta: float = 0.05, vel_lambda: float = 0.2):
        """
        Args:
            beta: Smoothness threshold for SmoothL1 (default 0.05)
            vel_lambda: Weight for velocity penalty (default 0.2)
        """
        super().__init__()
        self.beta = beta
        self.vel_lambda = vel_lambda

    def forward(self, pred_heights, target_heights, target_occlusion):
        """
        Compute trajectory loss with occlusion weighting.

        Args:
            pred_heights: (B, 32) predicted heights
            target_heights: (B, 32) target heights
            target_occlusion: (B, 32) occlusion values [0, 1]

        Returns:
            loss: scalar loss value
        """
        batch_size = pred_heights.size(0)

        # Compute per-frame spatial loss (Smooth L1)
        spatial_loss = F.smooth_l1_loss(pred_heights, target_heights,
                                        beta=self.beta, reduction='none')  # (B, 32)

        # Compute velocity penalty (L1 on frame-to-frame deltas)
        pred_vel = pred_heights[:, 1:] - pred_heights[:, :-1]  # (B, 31)
        target_vel = target_heights[:, 1:] - target_heights[:, :-1]  # (B, 31)
        vel_loss = torch.abs(pred_vel - target_vel)  # (B, 31)

        # Pad velocity loss to match spatial loss shape (B, 32)
        # First frame has no velocity penalty (no previous frame)
        vel_loss_padded = F.pad(vel_loss, (1, 0), value=0)  # (B, 32)

        # Compute occlusion weights (1 - occlusion)
        weights = 1 - target_occlusion  # (B, 32)

        # Combine spatial and velocity losses
        total_frame_loss = spatial_loss + self.vel_lambda * vel_loss_padded  # (B, 32)

        # Apply occlusion weighting and average
        weighted_loss = weights * total_frame_loss  # (B, 32)
        loss = weighted_loss.sum() / (batch_size * 32)

        return loss


class SmoothL1Loss(nn.Module):
    """
    Standard SmoothL1Loss (available in PyTorch, provided for completeness).
    """

    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=self.beta)
