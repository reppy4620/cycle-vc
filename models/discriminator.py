import torch.nn as nn

from .commons import GLU, Conv2dLayer, SimpleDecoder


class Discriminator(nn.Module):
    def __init__(self, base_channels=64, spectral_norm=True):
        super(Discriminator, self).__init__()

        conv = nn.Conv2d(1, base_channels * 2, kernel_size=3, padding=1)
        if spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        self.in_layer = nn.Sequential(
            conv,
            GLU(dim=1)
        )
        self.layers = nn.ModuleList([
            Conv2dLayer(base_channels, mag=4, kernel_size=3, stride=2, spectral_norm=spectral_norm),
            Conv2dLayer(base_channels * 2, mag=4, kernel_size=3, stride=2, spectral_norm=spectral_norm),
            Conv2dLayer(base_channels * 4, mag=4, kernel_size=3, stride=2, spectral_norm=spectral_norm),
            Conv2dLayer(base_channels * 8, mag=2, kernel_size=(1, 5), stride=1, padding=(0, 2), spectral_norm=spectral_norm),
        ])
        self.out = nn.Conv2d(base_channels * 8, 1, kernel_size=(1, 3), padding=(0, 1))

        self.decoder = SimpleDecoder(channels=base_channels * 8)

    def forward(self, x, need_recon=True):
        fms = list()
        x = x.unsqueeze(1)
        x = self.in_layer(x)
        for layer in self.layers:
            x = layer(x)
            fms.append(x)
        recon = None if not need_recon else self.decoder(x)
        x = self.out(x)
        x = x.squeeze(1)
        return x, fms, recon
