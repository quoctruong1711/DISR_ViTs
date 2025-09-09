"""CBENet with ViT backbone features.

This module extends the original CBENet by extracting global
representations with a Vision Transformer and fusing them with the
convolutional input. The ViT features are projected to three channels
and concatenated with the RGB input before being processed by the
FCDenseNet architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_backbone import ViTBackbone
from .CBENet import FCDenseNet


class ViTCBENet(nn.Module):
    """CBENet variant that incorporates ViT features."""

    def __init__(self, pretrained_vit: bool = True) -> None:
        super().__init__()
        # Vision Transformer backbone
        self.vit = ViTBackbone(pretrained=pretrained_vit)
        # Project high-dimensional ViT features to 3 channels
        self.proj = nn.Conv2d(768, 3, kernel_size=1)
        # Original CBENet expecting six-channel input (RGB + ViT features)
        self.cbenet = FCDenseNet(
            in_channels=6,
            down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4),
            bottleneck_layers=16,
            growth_rate=12,
            out_chans_first_conv=48,
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input RGB image of shape ``(B, 3, H, W)``.
        """
        vit_feat = self.vit(x)
        vit_feat = F.interpolate(
            vit_feat, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        vit_feat = self.proj(vit_feat)
        fused = torch.cat([x, vit_feat], dim=1)
        return self.cbenet(fused)


__all__ = ["ViTCBENet"]
