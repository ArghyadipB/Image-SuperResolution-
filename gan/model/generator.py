import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_activation=True,
        use_bn=True,
        **kwargs
    ):
        super().__init__()
        self.use_activation = use_activation
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x) if self.use_activation else x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = ConvBlock(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.block2 = ConvBlock(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=False,
        )

    def forward(self, x):
        return x + self.block2(self.block1(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super().__init__()

        self.initial = ConvBlock(
            in_channels,
            num_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            use_bn=False,
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv = ConvBlock(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=False,
        )

        # 🔥 UPSAMPLING (add more blocks for higher scale)
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            # Uncomment below for 4x:
            # UpsampleBlock(num_channels, scale_factor=2),
        )

        self.final = nn.Conv2d(
            num_channels, in_channels, kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.residuals(x1)
        x3 = self.conv(x2) + x1
        x4 = self.upsample(x3)
        return torch.sigmoid(self.final(x4))