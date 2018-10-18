import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features, hid_size, sizes):
        super().__init__()
        self.layers = []

        s_in = n_features

        for s_out in sizes:
            self.layers.append(nn.Linear(s_in, s_out))
            self.layers.append(nn.LeakyReLU())
            s_in = s_out

        self.layers.append(nn.Linear(s_in, hid_size))

        self.encode = nn.Sequential(
            *self.layers
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self, hid_size, n_features, sizes):
        super().__init__()
        self.layers = []
        s_in = hid_size

        for s_out in sizes:
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Linear(s_in, s_out))
            s_in = s_out

        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(s_in, n_features))

        self.decode = nn.Sequential(
            *self.layers
        )

    def forward(self, x):
        out = self.decode(x)
        return out
