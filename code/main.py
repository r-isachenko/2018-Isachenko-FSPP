# coding=utf-8

import torch
import torch.optim as optim
import torch.nn as nn

from trainer import Trainer
from models import Encoder, Decoder
from sampler import RandomSampler

N_FEATURES = 100
HID_SIZE = 10
ENCODER_SIZES = [50, 30, 20]
DECODER_SIZES = []
N_ITERS = 10000
BATCH_SIZE = 1024
LR = 10.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    sampler = RandomSampler(N_FEATURES)

    encoder = Encoder(
        N_FEATURES,
        HID_SIZE,
        ENCODER_SIZES
    ).to(device)

    decoder = Decoder(
        HID_SIZE,
        N_FEATURES,
        DECODER_SIZES
    ).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)

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


    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    print("encoder params:", get_n_params(encoder))
    print("decoder params:", get_n_params(decoder))

    print(encoder)
    print(decoder)

    trainer.train()
