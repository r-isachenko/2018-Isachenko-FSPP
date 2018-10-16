# coding=utf-8
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from models import Encoder, Decoder
from sampler import RandomSampler, binary2idx

N_FEATURES = 100
EMBED_SIZE = 20
HID_SIZE = 10
ENCODER_SIZES = [20]
DECODER_SIZES = [30]
N_ITERS = 10000
BATCH_SIZE = 1024
LR = 1.


class Trainer:
    def __init__(
            self,
            batch_size,
            n_iters,
            sampler,
            models,
            optimizers,
            criterion
        ):
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.sampler = sampler
        self.encoder = models[0]
        self.decoder = models[1]
        self.encoder_optimizer = optimizers[0]
        self.decoder_optimizer = optimizers[1]
        self.criterion = criterion

    def train(self):
        iterator = tqdm(range(self.n_iters))
        for n_iter in iterator:
            batch = self.sampler.generate(n_samples=self.batch_size)
            seq, offsets = self.batch2input(batch)
            encoding = self.encoder(seq, offsets)

            logits = self.decoder(encoding)
            gt = torch.tensor(batch, dtype=torch.float32)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.criterion(logits, gt)

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            iterator.set_postfix_str(f"loss={loss.item():.4f}")

    def batch2input(self, batch):
        batch = binary2idx(batch)
        lens = [0] + [len(sample) for sample in batch[:-1]]
        offsets = torch.tensor(np.cumsum(lens))
        seq = np.concatenate(batch)
        seq = torch.tensor(seq)
        return seq, offsets


if __name__ == "__main__":
    sampler = RandomSampler(N_FEATURES)

    encoder = Encoder(
        N_FEATURES,
        EMBED_SIZE,
        HID_SIZE,
        ENCODER_SIZES
    )

    decoder = Decoder(
        HID_SIZE,
        N_FEATURES,
        DECODER_SIZES
    )

    print(encoder)
    print(decoder)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR, weight_decay=1e-4)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR, weight_decay=1e-4)

    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        BATCH_SIZE,
        N_ITERS,
        sampler,
        (encoder, decoder),
        (encoder_optimizer, decoder_optimizer),
        criterion
    )

    trainer.train()