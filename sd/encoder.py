from torch.nn import functional as F
import torch
from torch import nn
from decoder import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # [b_size, channels, height, width] -> [b_size, 128, height, width]
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # [b_size, 128, height, width] -> [b_size, 128, height, width]
            VAEResidualBlock(128, 128),
            # [b_size, 128, height, width] -> [b_size, 128, height, width]
            VAEResidualBlock(128, 128),
            # [b_size, 128, height, width] -> [b_size, 128, height//2, width//2]
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # [b_size, 256, height//2, width//2] -> [b_size, 256, height//2, width//2]
            VAEResidualBlock(128, 256),
            # [b_size, 256, height//2, width//2] -> [b_size, 256, height//2, width//2]
            VAEResidualBlock(256, 256),
            # [b_size, 256, height//2, width//2] -> [b_size, 256, height//4, width//4]
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # [b_size, 256, height//4, width//4] -> [b_size, 512, height//4, width//4]
            VAEResidualBlock(256, 512),
            # [b_size, 512, height//4, width//4] -> [b_size, 512, height//4, width//4]
            VAEResidualBlock(512, 512),
            # [b_size, 512, height//4, width//4] -> [b_size, 512, height//8, width//8]
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # [b_size, 512, height//8, width//8] -> [b_size, 512, height//8, width//8]
            VAEResidualBlock(512, 512),
            # [b_size, 512, height//8, width//8] -> [b_size, 512, height//8, width//8]
            VAEResidualBlock(512, 512),
            # [b_size, 512, height//8, width//8] -> [b_size, 512, height//8, width//8]
            VAEResidualBlock(512, 512),
            # [b_size, 512, height//8, width//8] -> [b_size, 512, height//8, width//8]
            VAEAttentionBlock(512),
            # [b_size, 512, height//8, width//8] -> [b_size, 512, height//8, width//8]
            VAEResidualBlock(512, 512),
            # [b_size, 512, height//8, width//8] -> [b_size, 512, height//8, width//8]
            nn.GroupNorm(32, 512),
            # [b_size, 512, height//8, width//8] -> [b_size, 512, height//8, width//8]
            nn.SiLU(),
            # [b_size, 512, height//4, width//4] -> [b_size, 8, height//8, width//8]
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # [b_size, 8, height//4, width//4] -> [b_size, 8, height//8, width//8]
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: [b_size, channel, height, width]
        # noise: [b_size, out_channels, height//8, width//8]
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # [padding_left, padding_right, padding_top, padding_bottom]
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # [b_size, 8, height, height//8, width//8] -> two tensors of shape [b_size, 4, height, height//8, width//8]
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # [b_size, 4, height, height//8, width//8] -> [b_size, 4, height, height//8, width//8]
        log_variance = torch.clamp(log_variance, -30, 20)
        # [b_size, 4, height, height//8, width//8] -> [b_size, 4, height, height//8, width//8]
        variance = log_variance.exp()
        # [b_size, 4, height, height//8, width//8] -> [b_size, 4, height, height//8, width//8]
        stdev = variance.sqrt()

        # Z=N(0,1) -> N(mean, variance)
        # X = mean + stdev * Z
        x = mean + stdev * noise

        # scale the output by a constant -> original repo
        x *= 0.18215

        return x
