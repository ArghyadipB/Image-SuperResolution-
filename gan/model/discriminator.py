import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_bn=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        features = [64, 64, 128, 128, 256, 256, 512, 512]
        layers = []

        for i, f in enumerate(features):
            layers.append(
                ConvBlock(
                    in_channels,
                    f,
                    stride=1 if i % 2 == 0 else 2,
                    use_bn=(i != 0),  # 🔥 critical
                )
            )
            in_channels = f

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)