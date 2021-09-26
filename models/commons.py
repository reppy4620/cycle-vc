import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.glu(x, dim=self.dim)


class Conv1dLayer(nn.Module):
    def __init__(self, channels, mag, kernel_size=5, stride=1):
        super(Conv1dLayer, self).__init__()
        self.conv = nn.Conv1d(channels, channels * mag, kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.InstanceNorm1d(channels * mag)
        self.act = GLU(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Conv2dLayer(nn.Module):
    def __init__(self, channels, mag, kernel_size=5, stride=1, spectral_norm=False):
        super(Conv2dLayer, self).__init__()
        conv = nn.Conv2d(channels, channels * mag, kernel_size, stride=stride, padding=kernel_size // 2)
        if spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        self.conv = conv
        self.norm = nn.InstanceNorm2d(channels * mag)
        self.act = GLU(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class To1dLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 256, 1)
        self.norm = nn.InstanceNorm1d(256)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.conv(x)
        x = self.norm(x)
        return x


class To2dLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(256, channels, 1)
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = x.view(x.size(0), 256, -1, x.size(-1))
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = GLU(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels * 2, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm1d(channels * 2)
        self.act = GLU(dim=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return x


class Conv1dRelu(nn.Module):
    def __init__(self, in_channels, channels=128, kernel_size=5):
        super(Conv1dRelu, self).__init__()
        self.conv = nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class TFAN1d(nn.Module):
    def __init__(self, n_mel, channels, out_channels):
        super(TFAN1d, self).__init__()
        self.layers = nn.ModuleList([
            Conv1dRelu(
                in_channels=n_mel if i == 0 else channels,
                channels=channels
            ) for i in range(3)
        ])
        self.beta = nn.Conv1d(channels, out_channels, 1)
        self.gamma = nn.Conv1d(channels, out_channels, 1)

    def forward(self, x, y):
        y = F.interpolate(y, size=x.size(-1))
        for layer in self.layers:
            y = layer(y)
        x = x * self.gamma(y) + self.beta(y)
        return x


class ConvTFANLayer(nn.Module):
    def __init__(self, n_mel, in_channels, out_channels, tfan_channels, kernel_size):
        super(ConvTFANLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.tfan = TFAN1d(n_mel, tfan_channels, out_channels)
        self.act = GLU(dim=1)

    def forward(self, x, y):
        x = self.conv(x)
        x = self.tfan(x, y)
        x = self.act(x)
        return x
