"""
MLP prediction head for trajectory prediction.

Simple feed-forward network: D → 512 → ReLU → Dropout(0.2) → 32
"""

import torch
import torch.nn as nn


class TrajectoryHead(nn.Module):
    """
    MLP regression head for predicting 32 future frames of height.

    Takes backbone features and outputs 32 scalars (one per frame).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 32)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) backbone features

        Returns:
            (batch_size, 32) height predictions
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
