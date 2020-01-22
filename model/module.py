import torch

import torch.nn as nn
from torch.nn.utils import spectral_norm


class SNConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.nn = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.nn(x)


class SNLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            spectral_norm(nn.Linear(in_channels, out_channels)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.nn(x)
