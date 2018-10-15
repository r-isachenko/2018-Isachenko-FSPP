import numpy as np

class RandomSampler:
    def __init__(self, n_features):
        self.n_features = n_features

    def get_sample(self):
        return np.random.randint(
            low=0,
            high=2,
            size=self.n_features
        )

    def generate(self, n_samples):
        return [self.get_sample() for _ in range(n_samples)]


def binary2idx(x):
    if isinstance(x, np.ndarray):
        x = [x]
    return [np.where(x_)[0] for x_ in x]