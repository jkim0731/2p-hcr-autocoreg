"""S64 subgoal 3 — small 3D-CNN that embeds (B, 1, 16, 16, 16) patches to (B, D).

Architecture:
    Stem           : Conv3d(1 → 16, k=3, p=1)
    ResBlock3D × 3 : 16 → 16 (s=2) → 32 (s=2) → 64 (s=2)
                     each block = two 3×3×3 convs + InstanceNorm + ReLU,
                     with 1×1 stride-2 shortcut when channels or stride change.
    Head           : AdaptiveAvgPool3d(1) → Linear(64 → D)

Parameter count ~230 k at D=64. Small enough for CPU training at moderate
batch size. Used by C2 image-conditioned GNN (S64 subgoal 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_c, affine=True)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_c, affine=True)
        if stride != 1 or in_c != out_c:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1,
                          stride=stride, bias=False),
                nn.InstanceNorm3d(out_c, affine=True),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.relu(h + res)


class PatchEncoder(nn.Module):
    """Small 3D ResNet on (B, 1, 16, 16, 16) → (B, out_dim)."""

    def __init__(self, out_dim: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, 16, kernel_size=3,
                              padding=1, bias=False)
        self.stem_norm = nn.InstanceNorm3d(16, affine=True)
        self.stem_relu = nn.ReLU(inplace=True)

        self.block1 = ResBlock3D(16, 16, stride=2)  # 16³ → 8³
        self.block2 = ResBlock3D(16, 32, stride=2)  # 8³  → 4³
        self.block3 = ResBlock3D(32, 64, stride=2)  # 4³  → 2³

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem_relu(self.stem_norm(self.stem(x)))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.gap(h).flatten(1)
        return self.proj(h)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
