import torch
from model.module import SNConv2DLayer, SNLinearLayer
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            SNConv2DLayer(2, 15, 5),
            SNConv2DLayer(15, 25, 7),
            SNConv2DLayer(25, 40, 9),
            SNConv2DLayer(40, 50, 11)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layers = nn.Sequential(
            SNLinearLayer(50, 50),
            SNLinearLayer(50, 10),
            SNLinearLayer(10, 1),
        )

    def forward(self, x):
        o = self.conv_layers(x)
        o = self.gap(o)  # [batch_size, channels, 1, 1]
        o = o.reshape((o.shape[0], o.shape[1]))  # [batch_size, channels]
        o = self.linear_layers(o)  # [batch_size, 1]
        return o


if __name__ == '__main__':
    ipt = torch.rand(2, 2, 257, 120)
    model = Discriminator()
    opt = model(ipt)
    print(opt.shape)
