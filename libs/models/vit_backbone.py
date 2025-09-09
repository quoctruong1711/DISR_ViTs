"""Vision Transformer backbone for DISR_ViTs.

This module provides a simple wrapper around a ViT model from
``torchvision`` that exposes a CNN‑like feature map. The resulting tensor
can be integrated with the existing CBENet/BGShadowNet pipelines.

The implementation assumes that ``torchvision`` provides the
``vit_b_16`` model. If the current version of ``torchvision`` does not
include ViT models, importing :class:`ViTBackbone` will raise an
``ImportError`` at runtime.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:  # ``vit_b_16`` is available in torchvision >= 0.13
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception:  # pragma: no cover - handled at runtime
    vit_b_16 = None  # type: ignore
    ViT_B_16_Weights = None  # type: ignore


class ViTBackbone(nn.Module):
    """Backbone that extracts 2D feature maps using a Vision Transformer.

    Parameters
    ----------
    pretrained: bool
        If ``True`` (default), loads ImageNet pretrained weights when
        available. Otherwise the model is randomly initialized.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if vit_b_16 is None:
            raise ImportError(
                "torchvision>=0.13 with ViT support is required for ViTBackbone"
            )

        weights: Optional[ViT_B_16_Weights]
        if pretrained and ViT_B_16_Weights is not None:  # type: ignore[assignment]
            weights = ViT_B_16_Weights.DEFAULT
        else:
            weights = None

        self.vit = vit_b_16(weights=weights)  # type: ignore[call-arg]
        # Remove classification head to expose raw features
        self.vit.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute feature maps from an input tensor.

        The output tensor has shape ``(B, C, H, W)`` where ``H`` and ``W``
        correspond to the number of patches along each spatial dimension.
        """

        # ``_process_input`` converts the image to a patch sequence
        x = self.vit._process_input(x)
        # Encoder returns sequence with class token at index 0
        x = self.vit.encoder(x)
        x = x[:, 1:, :]  # remove class token

        n_patches = x.shape[1]
        h = w = int(n_patches ** 0.5)
        x = x.reshape(x.shape[0], h, w, x.shape[2]).permute(0, 3, 1, 2).contiguous()
        return x

__all__ = ["ViTBackbone"]
