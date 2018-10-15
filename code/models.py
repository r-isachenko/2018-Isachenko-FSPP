import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features, embed_size, hid_size, sizes):
        super().__init__()
        self.layers = []
        self.embedding = nn.EmbeddingBag(n_features, embed_size)

        s_in = embed_size

        for s_out in sizes:
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Linear(s_in, s_out))
            s_in = s_out

        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(s_in, hid_size))

        self.encode = nn.Sequential(
            *self.layers
        )

    def forward(self, x, offsets):
        embed = self.embedding(x, offsets)
        out = self.encode(embed)
        return out


class Decoder(nn.Module):
    def __init__(self, hid_size, n_features, sizes):
        super().__init__()
        self.layers = []
        s_in = hid_size

        for s_out in sizes:
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Linear(s_in, s_out))
            s_in = s_out

        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(s_in, n_features))

        self.decode = nn.Sequential(
            *self.layers
        )

    def forward(self, x):
        out = self.decode(x)
        return out
