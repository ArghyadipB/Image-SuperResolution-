"""
VGG-Based Perceptual Loss for Super-Resolution

This module implements a perceptual loss using a pretrained VGG19 network.
Instead of comparing images at the pixel level (e.g., L1 or MSE), this loss
measures differences in high-level feature representations extracted from
a deep convolutional network.
"""


import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# vgg loss class
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval()

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.loss = nn.MSELoss()

        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        return self.loss(self.vgg(x), self.vgg(y))