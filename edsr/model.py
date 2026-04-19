"""
Enhanced Deep Super-Resolution (EDSR) Model

This module implements the EDSR architecture for single-image super-resolution.
EDSR improves upon traditional residual networks by removing unnecessary
normalization layers and using residual scaling for stable training.
"""

import torch
import torch.nn as nn


# MEAN SHIFT
class MeanShift(nn.Conv2d):
    def __init__(self, sign=-1):
        super().__init__(3, 3, kernel_size=1)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        std = (1.0, 1.0, 1.0)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = torch.tensor(rgb_mean)

        if sign == -1:
            self.bias.data *= -1

        for p in self.parameters():
            p.requires_grad = False


# RESIDUAL BLOCK (with scaling)
class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.block(x) * self.res_scale


# UPSAMPLER (PixelShuffle)
class Upsampler(nn.Sequential):
    def __init__(self, scale, channels):
        m = []

        if scale == 2:
            m.append(nn.Conv2d(channels, channels * 4, 3, 1, 1))
            m.append(nn.PixelShuffle(2))

        elif scale == 4:
            for _ in range(2):
                m.append(nn.Conv2d(channels, channels * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))

        else:
            raise ValueError("Scale must be 2 or 4")

        super().__init__(*m)


# EDSR MODEL
class EDSR(nn.Module):
    def __init__(self, scale=2, n_resblocks=16, n_feats=64, res_scale=0.1):
        super().__init__()

        # mean normalization
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        # head
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)

        # body
        self.body = nn.Sequential(
            *[ResidualBlock(n_feats, res_scale) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        # tail (upsampling)
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        x = self.add_mean(x)

        return torch.clamp(x, 0.0, 1.0)