import torch
from torch.utils.data import Sampler
from typing import Iterator, Sequence
import numpy as np

class WeightedRandomSampler(Sampler[int]):

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))

        self.weights = weights_tensor
        self.weights_numpy = weights
        self.p = weights/weights.sum()
        self.weight_shape = len(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:

        rand_tensor = np.random.choice(self.weight_shape, self.num_samples, p=self.p)
        np.random.permutation
        rand_tensor = torch.as_tensor(rand_tensor, dtype=torch.long)
        # rand_tensor = (self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples