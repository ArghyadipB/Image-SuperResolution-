"""
Loss Functions for Super-Resolution (Perceptual + Edge)

This module defines loss functions used to train super-resolution models,
focusing on both perceptual quality and structural sharpness.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

# vgg los and edge loss
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval()

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.loss = nn.MSELoss()

        self.register_buffer(
            "mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        )

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return self.loss(self.vgg(x), self.vgg(y))


def edge_loss(sr, hr):
    sobel_x = torch.tensor(
        [[1,0,-1],[2,0,-2],[1,0,-1]],
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(sr.device)

    sobel_y = sobel_x.permute(0,1,3,2)

    sr_gray = sr.mean(dim=1, keepdim=True)
    hr_gray = hr.mean(dim=1, keepdim=True)

    sr_edge = F.conv2d(sr_gray, sobel_x, padding=1) + \
              F.conv2d(sr_gray, sobel_y, padding=1)

    hr_edge = F.conv2d(hr_gray, sobel_x, padding=1) + \
              F.conv2d(hr_gray, sobel_y, padding=1)

    return F.l1_loss(sr_edge, hr_edge)