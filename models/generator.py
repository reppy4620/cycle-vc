import torch.nn as nn

from .commons import GLU, Conv2dLayer, To1dLayer, To2dLayer, UpsampleLayer, ResidualBlock


class Generator(nn.Module):
    def __init__(self, n_mel=80, base_channels=64, n_residuals=6):
        super(Generator, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(1, base_channels * 2, kernel_size=(5, 15), padding=(2, 7)),
            GLU(dim=1)
        )
        self.layer1 = Conv2dLayer(base_channels, mag=4, stride=2)
        self.layer2 = Conv2dLayer(base_channels * 2, mag=4, stride=2)

        self.to_1d_layer = To1dLayer(base_channels * 4 * (n_mel // 4))

        self.residuals = nn.ModuleList([
            ResidualBlock(
                base_channels * 4,
                kernel_size=3
            ) for _ in range(n_residuals)
        ])

        self.to_2d_layer = To2dLayer(base_channels * 4 * (n_mel // 4))

        self.layer3 = UpsampleLayer(base_channels * 4, base_channels * 16)
        self.layer4 = UpsampleLayer(base_channels * 2, base_channels * 8)
        self.out = nn.Conv2d(base_channels, 1, kernel_size=(5, 15), padding=(2, 7))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.in_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.to_1d_layer(x)

        for layer in self.residuals:
            x = layer(x)

        x = self.to_2d_layer(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)
        x = x.squeeze(1)
        return x
