"""Hypernetwork architecture for MEND (Gradient Decomposition)."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientFactorTransform(nn.Module):
    """Small MLP transforming input/output gradient factors into localized updates."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_dim),
        )
        # Residual scaling initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around the transformation
        return x + self.net(x)


class MENDHypernetwork(nn.Module):
    """Gradient Decomposition Network for a linear layer W (out_dim x in_dim)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.transform_u = GradientFactorTransform(in_dim, hidden_dim)
        self.transform_delta = GradientFactorTransform(out_dim, hidden_dim)

    def forward(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Decomposes gradient factors and synthesizes localized weight delta.
        
        u: [in_dim] activation factor
        delta: [out_dim] gradient factor
        Returns:
            delta_W: [out_dim, in_dim]
        """
        u_tilde = self.transform_u(u)
        delta_tilde = self.transform_delta(delta)
        delta_W = torch.outer(delta_tilde, u_tilde)
        return delta_W
