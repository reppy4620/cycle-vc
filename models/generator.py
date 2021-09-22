import torch.nn as nn

from .commons import GLU, Conv1dLayer, ResidualBlock, ConvTFANLayer


class Generator(nn.Module):
    def __init__(self, n_mel=80, base_channels=64, n_residuals=6):
        super(Generator, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv1d(n_mel, base_channels * 2, kernel_size=15, padding=7),
            GLU(dim=1)
        )
        self.layer1 = Conv1dLayer(base_channels, mag=4)
        self.layer2 = Conv1dLayer(base_channels * 2, mag=2)

        self.residuals = nn.ModuleList([
            ResidualBlock(
                base_channels * 2,
                kernel_size=3
            ) for _ in range(n_residuals)
        ])

        self.layer3 = ConvTFANLayer(
            n_mel=n_mel,
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            tfan_channels=base_channels * 2,
            kernel_size=5
        )
        self.layer4 = ConvTFANLayer(
            n_mel=n_mel,
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            tfan_channels=base_channels * 2,
            kernel_size=5
        )
        self.out = nn.Conv1d(base_channels * 2, n_mel, 1)

    def forward(self, x, y):
        x = self.in_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)

        for layer in self.residuals:
            x = layer(x)

        x = self.layer3(x, y)
        x = self.layer4(x, y)
        x = self.out(x)
        return x