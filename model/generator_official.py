import torch

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layer = nn.LSTM(input_size=257, hidden_size=200, num_layers=2, bidirectional=True, batch_first=True)
        self.linear_layer_1 = nn.Sequential(
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Dropout(0.05)
        )
        self.linear_layer_2 = nn.Sequential(
            nn.Linear(300, 257),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 1, 3, 2).squeeze(1)
        o, _ = self.lstm_layer(x)
        o = self.linear_layer_1(o)
        o = self.linear_layer_2(o)
        o = o.permute(0, 2, 1).unsqueeze(1)
        return o


if __name__ == '__main__':
    ipt = torch.rand(2, 1, 257, 120)
    model = Generator()
    opt = model(ipt)
    print(opt.shape)
