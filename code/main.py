# coding=utf-8

import torch
import torch.optim as optim
import torch.nn as nn

from trainer import Trainer
from models import Encoder, Decoder
from sampler import RandomSampler

N_FEATURES = 100
EMBED_SIZE = 10
HID_SIZE = 30
ENCODER_SIZES = []
DECODER_SIZES = [50]
N_ITERS = 10000
BATCH_SIZE = 1024
LR = 10.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    sampler = RandomSampler(N_FEATURES)

    encoder = Encoder(
        N_FEATURES,
        EMBED_SIZE,
        HID_SIZE,
        ENCODER_SIZES
    ).to(device)

    decoder = Decoder(
        HID_SIZE,
        N_FEATURES,
        DECODER_SIZES
    ).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR, weight_decay=1e-4)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR, weight_decay=1e-4)

    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=500, gamma=0.9)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=500, gamma=0.9)

    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        BATCH_SIZE,
        N_ITERS,
        sampler,
        (encoder, decoder),
        (encoder_optimizer, decoder_optimizer),
        (encoder_scheduler, decoder_scheduler),
        criterion,
        device
    )

    trainer.train()
