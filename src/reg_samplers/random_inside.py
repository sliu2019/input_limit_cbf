import torch
import IPython
import numpy as np
import sys

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

class RandomInsideRegSampler():
    """
    candidate_samples uniformly in h's zero sublevel set
    """
    # Note: this is not batch compliant.

    def __init__(self, x_lim, device, logger, n_samples=250):
        vars = locals()  # dict of local names
        self.__dict__.update(vars)  # __dict__ holds and object's attributes
        del self.__dict__["self"]  # don't need `self`

        self.x_dim = x_lim.shape[0]

        self.x_lim_interval_sizes = np.reshape(x_lim[:, 1] - x_lim[:, 0], (1, self.x_dim))
        self.bs = 100 # for evaluating on phi_fn

    def get_samples(self, phi_fn):
        # TODO: assuming phi_fn(x)[:, 0] is h(x)!!!! If it is not, this will not work
        # print("inside RandomInside sampler, get_ssamples")
        # IPython.embed()

        # Define some variables
        samples = torch.empty((0, self.x_dim), device=self.device)

        n_samp_found = 0
        i = 0
        while n_samp_found < self.n_samples:
            # print(i)
            # Sample in box
            candidate_samples_numpy = np.random.uniform(size=(self.bs, self.x_dim))*self.x_lim_interval_sizes + self.x_lim[:, [0]].T
            candidate_samples_torch = torch.from_numpy(candidate_samples_numpy.astype("float32")).to(self.device)

            phi_vals = phi_fn(candidate_samples_torch)
            # TODO: bug
            # max_phi_vals = torch.max(phi_vals, dim=1)[0]
            # ind = torch.nonzero(max_phi_vals <= 0).flatten()
            h_vals = phi_vals[:, 0]
            ind = torch.nonzero(h_vals <= 0).flatten()

            # Save good candidate_samples
            samples_inside = candidate_samples_torch[ind]
            samples = torch.cat((samples, samples_inside), dim=0)

            n_samp_found += len(ind)
            i += 1

        # Could be more than self.n_samples currently; truncate to exactly self.n_samples
        samples = samples[:self.n_samples]

        return samples
