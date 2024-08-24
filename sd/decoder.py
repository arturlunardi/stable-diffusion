from torch.nn import functional as F
import torch
from torch import nn
from attention import SelfAttention


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b_size, features, height, width]
        residue = x

        n, c, h, w = x.shape
        # [b_size, features, height, width] -> [b_size, features, height * width]
        x = x.view(n, c, h * w)
        # [b_size, features, height * width] -> [b_size, height * width, features]
        x = x.transpose(-1, -2)

        # [b_size, height * width, features] -> [b_size, height * width, features]
        x = self.attention(x)
        # [b_size, height * width, features] -> [b_size, features, height * width]
        x = x.transpose(-1, -2)
        # [b_size, features, height * width] -> [b_size, features, height, width]
        x = x.view(n, c, h, w)

        x += residue
        return x


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b_size, in_channels, height, width]
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            # [b_size, 512, height / 8, width / 8]
            VAEResidualBlock(512, 512),
            # [b_size, 512, height / 8, width / 8] -> [b_size, 512, height / 4, width / 4]
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            # [b_size, 512, height / 4, width / 4] -> [b_size, 512, height / 2, width / 2]
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            # [b_size, 256, height / 2, width / 2] -> [b_size, 256, height, width]
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # [b_size, 128, height, width] -> [b_size, 3, height, width]
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b_size, 4, height/8, width/8]
        x /= 0.18215

        for module in self:
            x = module(x)

        # [b_size, 3, height, width]
        return x
