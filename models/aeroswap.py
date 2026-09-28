"""Compact sky segmentation network for AeroSwap.

Inputs are RGB float tensors in [0, 1]. The ImageNet normalization needed by
the pretrained MobileNetV3-Small encoder lives inside the model, so an exported
TorchScript model can be passed directly to eval.aeroswap_eval.evaluate().
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=[skip.size(2), skip.size(3)], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((x, skip), dim=1))


class AeroSwapNet(nn.Module):
    """MobileNetV3-Small encoder with U-Net style skip-connected decoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = mobilenet_v3_small(weights=weights).features
        # The pinned torchvision 0.29 encoder has strides 2, 4, 8, 16, 32 at
        # these boundaries and channels 16, 16, 24, 48, 576 respectively.
        self.enc2 = nn.Sequential(*features[:1])
        self.enc4 = nn.Sequential(*features[1:2])
        self.enc8 = nn.Sequential(*features[2:4])
        self.enc16 = nn.Sequential(*features[4:9])
        self.enc32 = nn.Sequential(*features[9:13])

        self.up16 = UpBlock(576, 48, 128)
        self.up8 = UpBlock(128, 24, 64)
        self.up4 = UpBlock(64, 16, 32)
        self.up2 = UpBlock(32, 16, 16)
        self.final = ConvBlock(16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        x2 = self.enc2(x)
        x4 = self.enc4(x2)
        x8 = self.enc8(x4)
        x16 = self.enc16(x8)
        x32 = self.enc32(x16)
        x = self.up16(x32, x16)
        x = self.up8(x, x8)
        x = self.up4(x, x4)
        x = self.up2(x, x2)
        x = F.interpolate(x, size=[image.size(2), image.size(3)], mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(self.final(x)))
