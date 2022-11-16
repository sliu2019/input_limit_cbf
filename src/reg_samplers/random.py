import torch
import IPython
import numpy as np
import sys

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

class RandomRegSampler():
    """
    Samples uniformly in state domain
    """
    # Note: this is not batch compliant.

    def __init__(self, x_lim, device, logger, n_samples=250):
        vars = locals()  # dict of local names
        self.__dict__.update(vars)  # __dict__ holds and object's attributes
        del self.__dict__["self"]  # don't need `self`

        self.x_dim = x_lim.shape[0]

        self.x_lim_interval_sizes = np.reshape(x_lim[:, 1] - x_lim[:, 0], (1, self.x_dim))

    def get_samples(self, phi_fn):
        # print("Random, get_samples")
        # print("Also check init")
        # IPython.embed()
        samp_numpy = np.random.uniform(size=(self.n_samples, self.x_dim))*self.x_lim_interval_sizes + self.x_lim[:, [0]].T
        samp_torch = torch.from_numpy(samp_numpy.astype("float32")).to(self.device)
        return samp_torch

