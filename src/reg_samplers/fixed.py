import torch
import IPython
import numpy as np
import sys

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

class FixedRegSampler():
    """
    Keeps points on the invariant set boundary to be used in the regularization term
    """
    # Note: this is not batch compliant.

    def __init__(self, x_lim, device, logger, n_samples=250, samples=None):
        vars = locals()  # dict of local names
        self.__dict__.update(vars)  # __dict__ holds and object's attributes
        del self.__dict__["self"]  # don't need `self`
        self.x_dim = x_lim.shape[0]
        self.x_lim_interval_sizes = np.reshape(x_lim[:, 1] - x_lim[:, 0], (1, self.x_dim))

        # print("Fixed, get_samples")
        # print("Also check init")
        # IPython.embed()

        if self.samples is None:
            samp_numpy = np.random.uniform(size=(self.n_samples, self.x_dim)) * self.x_lim_interval_sizes + self.x_lim[:, [0]].T
            samp_torch = torch.from_numpy(samp_numpy.astype("float32")).to(self.device)
            self.samples = samp_torch

    def get_samples(self, phi_fn):
        return self.samples

