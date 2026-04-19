"""
UNet-Based Super-Resolution Model with PixelShuffle Upsampling

This module implements a UNet architecture adapted for image super-resolution.
It combines encoder–decoder skip connections with a PixelShuffle-based output
layer to efficiently upscale low-resolution images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# DOUBLE CONV BLOCK
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


# UNET WITH PIXELSHUFFLE
class UNET(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()

        self.scale = scale_factor
        features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        in_c = 3
        for f in features:
            self.downs.append(DoubleConv(in_c, f))
            in_c = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        # 🔥 PIXELSHUFFLE HEAD
        self.final = nn.Sequential(
            nn.Conv2d(features[0], 3 * (self.scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale)
        )

    def forward(self, x):
        skips = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skips = skips[::-1]

        # Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = self.ups[i + 1](torch.cat([skip, x], dim=1))

        # OUTPUT (UPSCALED)
        return torch.sigmoid(self.final(x))