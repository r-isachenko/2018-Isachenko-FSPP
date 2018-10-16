from tqdm import tqdm
import numpy as np

import torch

from sampler import binary2idx


class Trainer:
    def __init__(
            self,
            batch_size,
            n_iters,
            sampler,
            models,
            optimizers,
            schedulers,
            criterion,
            device
    ):
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.sampler = sampler
        self.encoder = models[0]
        self.decoder = models[1]
        self.encoder_optimizer = optimizers[0]
        self.decoder_optimizer = optimizers[1]
        self.encoder_scheduler = schedulers[0]
        self.decoder_scheduler = schedulers[1]
        self.criterion = criterion
        self.device = device

    def train(self):
        iterator = tqdm(range(self.n_iters))
        for n_iter in iterator:
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()
            batch = self.sampler.generate(n_samples=self.batch_size)
            seq, offsets = self.batch2input(batch)
            encoding = self.encoder(seq, offsets)

            gt = torch.tensor(batch, dtype=torch.float32).to(self.device)

            logits = self.decoder(encoding)
            hamming = torch.mean(torch.lt(logits * (gt - 0.5), 0).to(self.device, dtype=torch.float32))

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.criterion(logits, gt)

            loss.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            iterator.set_postfix_str(f"loss={loss.item():.4f}, hamming={hamming.item():.4f}")

    def batch2input(self, batch):
        batch = binary2idx(batch)
        lens = [0] + [len(sample) for sample in batch[:-1]]
        offsets = torch.tensor(np.cumsum(lens).astype(np.float), dtype=torch.long).to(self.device)
        seq = np.concatenate(batch)
        seq = torch.tensor(seq).to(self.device)
        return seq, offsets