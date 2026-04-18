import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Use pretrained VGG19 features
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval()

        # Freeze VGG parameters
        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.loss = nn.MSELoss()

        # Register mean and std as buffers (auto moves with device)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x, y):
        # Normalize inputs to match VGG training distribution
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        return self.loss(self.vgg(x), self.vgg(y))