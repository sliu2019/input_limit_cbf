import torch
import IPython
import numpy as np
import sys

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *

class BoundaryRegSampler():
    """
    Keeps points on the invariant set boundary to be used in the regularization term
    """
    # Note: this is not batch compliant.

    def __init__(self, x_lim, device, logger, n_samples=250, \
                 projection_tolerance=1e-1, projection_lr=1e-2, projection_time_limit=3, verbose=False):
        vars = locals()  # dict of local names
        self.__dict__.update(vars)  # __dict__ holds and object's attributes
        del self.__dict__["self"]  # don't need `self`

        self.x_lim = torch.tensor(x_lim).to(device)
        self.x_dim = self.x_lim.shape[0]

        # Compute 2n facets volume of n-dim hypercube (actually n facets because they come in pairs)
        x_lim_interval_sizes = self.x_lim[:, 1] - self.x_lim[:, 0]
        x_lim_interval_sizes = x_lim_interval_sizes.view(1, -1)
        tiled = x_lim_interval_sizes.repeat(self.x_dim, 1)
        tiled = tiled - torch.eye(self.x_dim).to(self.device)*x_lim_interval_sizes + torch.eye(self.x_dim).to(device)
        vols = torch.prod(tiled, axis=1)
        vols = vols/torch.sum(vols)
        self.vols = vols.detach().cpu().numpy() # numpy
        self.hypercube_vol = torch.prod(x_lim_interval_sizes) # tensor const

        # For warmstart
        self.X_saved = None

        # print("NEED TO PROOFREAD THIS IMPLEMENTATION REGSAMPLEKEEPER BEFORE USING IT")
        # sys.exit(0)

    def _project(self, phi_fn, x):
        # NOTE: it doesn't matter much for reg for the points to be exactly on the boundary!

        # Recommend GD instead of line search for this one, since the objective is a max...
        # Until convergence
        i = 0
        t1 = time.perf_counter()

        x_list = list(x)
        x_list = [x_mem.view(-1, self.x_dim) for x_mem in x_list]
        for x_mem in x_list:
            x_mem.requires_grad = True
        proj_opt = optim.Adam(x_list, lr=self.projection_lr)

        while True:
            proj_opt.zero_grad()
            loss = torch.sum(torch.abs(torch.max(phi_fn(torch.cat(x_list), grad_x=True), axis=1)[0])) # TODO
            loss.backward()
            proj_opt.step()

            i += 1
            t_now = time.perf_counter()
            if torch.max(loss) < self.projection_tolerance:
                break
            elif (t_now - t1) > self.projection_time_limit:
                print("Reg: reprojection exited on timeout, max dist from =0 boundary: ", loss.item())
                # print("Attack: reprojection exited on timeout, max dist from =0 boundary: ", torch.max(loss).item())
                break

        for x_mem in x_list:
            x_mem.requires_grad = False
        rv_x = torch.cat(x_list)

        if self.verbose:
            # if torch.max(loss) < self.projection_tolerance:
            #     print("Yes, on manifold")
            # else:
            if torch.max(loss) > self.projection_tolerance:
                print("Not on manifold, %f" % (torch.max(loss).item()))
        return rv_x

    def _sample_in_cube(self):
        """
        Samples uniformly in state space hypercube
        Returns 1 sample
        """
        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        return sample

    def _sample_on_cube(self):
        """
        Samples uniformly on state space hypercube
        Returns 1 sample
        """
        # https://math.stackexchange.com/questions/2687807/uniquely-identify-hypercube-faces
        which_facet_pair = np.random.choice(np.arange(self.x_dim), p=self.vols)
        which_facet = np.random.choice([0, 1])

        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        sample[which_facet_pair] = self.x_lim[which_facet_pair, which_facet]
        return sample

    def _intersect_segment_with_manifold(self, p1, p2, phi_fn, rtol=1e-5, atol=1e-3):
        """
        Atol? Reltol?
        """
        # IPython.embed()
        diff = p2-p1

        left_weight = 0.0
        right_weight = 1.0
        # left_val = phi_fn(p1.view(1, -1))[0, -1] # TODO
        # right_val = phi_fn(p2.view(1, -1))[0, -1] # TODO

        left_val = torch.max(phi_fn(p1.view(1, -1))).item()
        right_val = torch.max(phi_fn(p2.view(1, -1))).item()

        left_sign = np.sign(left_val)
        right_sign = np.sign(right_val)

        if left_sign*right_sign > 0:
            return None

        t0 = time.perf_counter()
        while True:
            mid_weight = (left_weight + right_weight)/2.0
            mid_point = p1 + mid_weight*diff

            # mid_val = phi_fn(mid_point.view(1, -1))[0, -1] # TODO
            mid_val = torch.max(phi_fn(mid_point.view(1, -1))).item()

            mid_sign = np.sign(mid_val)
            if mid_sign*left_sign < 0:
                # go to the left side
                right_weight = mid_weight
                right_val = mid_val
            elif mid_sign*right_sign <= 0:
                left_weight = mid_weight
                left_val = mid_val

            # Use this approach or the one below to prevent infinite loops
            # Approach #1: previously used for discontinuous phi, but we shouldn't have discont. phi
            # if np.abs(left_weight - right_weight) < 1e-3:
            #     intersection_point = p1 + left_weight*diff
            #     break
            if max(abs(left_val), abs(right_val)) < self.projection_tolerance:
                intersection_point = p1 + left_weight*diff
                break
            t1 = time.perf_counter()
            if (t1-t0)>7:
                # This clause is necessary for non-differentiable, continuous points (abrupt change)
                print("Something is wrong in projection for RegSampleKeeper")
                print(left_val, right_val)
                # print(torch.abs(left_val - right_val))
                print(left_weight, right_weight)
                print("p1:", p1)
                print("p2:", p2)
                # print(torch.abs(left_val - right_val))
                # print(left_weight, right_weight)
                # left_point = p1 + left_weight * diff
                # right_point = p1 + right_weight * diff
                # print(left_point, right_point)
                # print(left_val, right_val)
                # print(mid_val, mid_point, mid_sign)
                # IPython.embed()
                return None

        return intersection_point

    def _sample_invariant_set_boundary(self, phi_fn):
        """
        Returns torch array of size (self.n_samples, self.x_dim)
        """
        # print("In sample_invariant_set_boundary")
        # IPython.embed()

        # Everything done in torch
        samples = []
        n_remaining_to_sample = self.n_samples

        center = self._sample_in_cube()
        n_segments_sampled = 0
        while n_remaining_to_sample > 0:
            # print(n_remaining_to_sample)
            outer = self._sample_on_cube()

            intersection = self._intersect_segment_with_manifold(center, outer, phi_fn)
            # valid = False
            if intersection is not None:
                samples.append(intersection.view(1, -1))
                n_remaining_to_sample -= 1
                # valid = True

            # if not valid:
            center = self._sample_in_cube()
            n_segments_sampled += 1
            # self.logger.info("%i segments" % n_segments_sampled)

        samples = torch.cat(samples, dim=0)
        # self.logger.info("Done with sampling points on the boundary...")
        return samples

    def get_samples(self, phi_fn):
        if self.X_saved is None:
            # print("sampling for the first time")
            self.X_saved = self._sample_invariant_set_boundary(phi_fn)
        else:
            # print("Updating samples")
            self.X_saved = self._project(phi_fn, self.X_saved) # reproject, since phi changed

        # IPython.embed()
        return self.X_saved
