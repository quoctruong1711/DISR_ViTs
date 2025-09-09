"""BGShadowNet variants with integrated ViT features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_backbone import ViTBackbone
from .stageI import FCDenseNet as StageIFCDenseNet
from .stageII import FCDenseNet as StageIIFCDenseNet


class ViTBGShadowNet1(StageIFCDenseNet):
    """First stage of BGShadowNet enhanced with ViT features."""

    def __init__(self, pretrained_vit: bool = True) -> None:
        super().__init__(
            in_channels=6,
            down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4),
            bottleneck_layers=16,
            growth_rate=12,
            out_chans_first_conv=48,
        )
        self.vit = ViTBackbone(pretrained=pretrained_vit)
        self.proj = nn.Conv2d(768, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, featureMaps):
        vit_feat = self.vit(x)
        vit_feat = F.interpolate(
            vit_feat, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        vit_feat = self.proj(vit_feat)
        x = torch.cat([x, vit_feat], dim=1)
        return super().forward(x, featureMaps)


class ViTBGShadowNet2(StageIIFCDenseNet):
    """Second stage of BGShadowNet enhanced with ViT features."""

    def __init__(self, pretrained_vit: bool = True) -> None:
        super().__init__(
            in_channels=9,
            down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4),
            bottleneck_layers=12,
            growth_rate=12,
            out_chans_first_conv=48,
        )
        self.vit = ViTBackbone(pretrained=pretrained_vit)
        self.proj = nn.Conv2d(768, 3, kernel_size=1)

    def forward(self, confuse_result, background, shadow_img, featureMaps):
        vit_feat = self.vit(shadow_img)
        vit_feat = F.interpolate(
            vit_feat, size=shadow_img.shape[-2:], mode="bilinear", align_corners=False
        )
        vit_feat = self.proj(vit_feat)
        x = torch.cat([confuse_result, shadow_img, vit_feat], dim=1)

        background_feature = []
        back1 = self.Cv0(background)
        background_feature.append(back1)
        back2 = self.Cv1(back1)
        background_feature.append(back2)
        back3 = self.Cv2(back2)
        background_feature.append(back3)
        back4 = self.Cv3(back3)
        background_feature.append(back4)
        back5 = self.Cv4(back4)
        background_feature.append(back5)

        out = self.firstconv(x)
        DEModuleFirst = self.DEModulefirstConv(x)
        skip_connections = []
        newFeatureMap = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            background_featuremap = background_feature[i]
            skip = torch.cat((out, background_featuremap), 1)
            att = getattr(self, f"att{i}")(skip)
            skip = skip * att
            skip_connections.append(skip)
            newFeatureMap.append(out)
            out = self.transDownBlocks[i](out)
        DEModuleinput = torch.cat([DEModuleFirst, skip_connections[1]], dim=1)
        DEModuleresult = self.DEModule(DEModuleinput)
        skip_connections[1] = torch.cat([skip_connections[1], DEModuleresult], dim=1)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            featureMap = featureMaps.pop()
            out = self.transUpBlocks[i](out, skip, featureMap)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        return out, newFeatureMap


__all__ = ["ViTBGShadowNet1", "ViTBGShadowNet2"]
