import torch
import IPython
import numpy as np

from torch import nn
from torch.autograd import grad
import torch.optim as optim
import time
from src.utils import *
import IPython
# import multiprocessing as mp
import math
import torch.multiprocessing as mp


class BatchWarmstartAttacker():

    def __init__(self, x_lim, device, logger, n_samples=60, \
                 stopping_condition="n_steps", max_n_steps=50, early_stopping_min_delta=1e-3, early_stopping_patience=50,\
                 lr=1e-3, \
                 p_reuse=0.7,\
                 projection_tolerance=1e-1, projection_lr=1e-2, projection_time_limit=3.0, verbose=False, train_attacker_use_n_step_schedule=False,\
                 boundary_sampling_speedup_method="sequential", boundary_sampling_method="gaussian", gaussian_t=1.0):

        vars = locals()  # dict of local names
        self.__dict__.update(vars)  # __dict__ holds and object's attributes
        del self.__dict__["self"]  # don't need `self`

        assert stopping_condition in ["n_steps", "early_stopping"]

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
        self.obj_vals_saved = None

        # For multiproc
        self.n_gpu = torch.cuda.device_count()
        if self.boundary_sampling_speedup_method == "gpu_parallelized":
            self.pool = mp.Pool(self.n_gpu) # torch pool
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                print("Couldn't set start_method as spawn, in init")
                IPython.embed()

    def __getstate__(self): # can't pickle pool object; creating threads will pickle self
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def _project(self, phi_fn, X, projection_n_grad_steps=None):
        """
        GPU batched
        GD-based projection
        With timeout
        """
        # Until convergence
        i = 0
        t1 = time.perf_counter()

        X_list = list(X)
        X_list = [X_mem.view(-1, self.x_dim) for X_mem in X_list] # TODO
        for X_mem in X_list:
            X_mem.requires_grad = True
        proj_opt = optim.Adam(X_list, lr=self.projection_lr)

        while True:
            proj_opt.zero_grad()

            # loss = torch.sum(torch.abs(surface_fn(torch.cat(X_list), grad_x=True)))
            loss = torch.sum(torch.abs(phi_fn(torch.cat(X_list), grad_x=True)[:, -1]))

            loss.backward()
            proj_opt.step()

            i += 1
            # print(i)
            t_now = time.perf_counter()
            if torch.max(loss) < self.projection_tolerance:
                # if self.verbose:
                #     print("reprojection exited before timeout in %i steps" % i)
                break

            if projection_n_grad_steps is not None: # use step number limit
                if i == projection_n_grad_steps:
                    break
            else: # else, use time limit
                if (t_now - t1) > self.projection_time_limit:
                    # print("reprojection exited on timeout")
                    print("Attack: reprojection exited on timeout, max dist from =0 boundary: ", torch.max(loss).item())
                    break

            # print((t_now - t1), torch.max(loss))

        for X_mem in X_list:
            X_mem.requires_grad = False
        rv_X = torch.cat(X_list)

        if self.verbose:
            # if torch.max(loss) < self.projection_tolerance:
            #     print("Yes, on manifold")
            # else:
            if torch.max(loss) > self.projection_tolerance:
                print("Not on manifold, %f" % (torch.max(loss).item()))
        # IPython.embed()
        return rv_X

    def _step(self, objective_fn, phi_fn, X):
        # It makes less sense to use an adaptive LR method here, if you think about it
        t0_step = time.perf_counter()

        X_batch = X.view(-1, self.x_dim)
        X_batch.requires_grad = True

        obj_val = -objective_fn(X_batch) # maximizing
        obj_grad = grad([torch.sum(obj_val)], X_batch)[0]

        # normal_to_manifold = grad([torch.sum(surface_fn(X_batch))], X_batch)[0]
        normal_to_manifold = grad([torch.sum(phi_fn(X_batch)[:, -1])], X_batch)[0]

        normal_to_manifold = normal_to_manifold/torch.norm(normal_to_manifold, dim=1)[:, None] # normalize
        X_batch.requires_grad = False
        weights = obj_grad.unsqueeze(1).bmm(normal_to_manifold.unsqueeze(2))[:, 0]
        proj_obj_grad = obj_grad - weights*normal_to_manifold

        # Take a step
        X_new = X - self.lr*proj_obj_grad
        tf_grad_step = time.perf_counter()

        # dist_before_proj = torch.mean(torch.abs(surface_fn(X_new)))
        dist_before_proj = torch.mean(torch.abs(phi_fn(X_new)[:,-1]))
        X_new = self._project(phi_fn, X_new)
        # dist_after_proj = torch.mean(torch.abs(surface_fn(X_new)))
        dist_after_proj = torch.mean(torch.abs(phi_fn(X_new)[:,-1]))

        tf_reproject = time.perf_counter()

        # Wrap-around in state domain
        X_new = torch.minimum(torch.maximum(X_new, self.x_lim[:, 0]), self.x_lim[:, 1])
        dist_diff_after_proj = (dist_after_proj-dist_before_proj).detach().cpu().numpy()
        debug_dict = {"t_grad_step": (tf_grad_step-t0_step), "t_reproject": (tf_reproject-tf_grad_step), "dist_diff_after_proj": dist_diff_after_proj}

        # print("Inside _step, line 242")
        # IPython.embed()
        return X_new, debug_dict

    def _sample_in_cube(self, random_seed=None):
        """
        Samples uniformly in state space hypercube
        Returns 1 sample
        """
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        return sample

    def _sample_on_cube(self, random_seed=None):
        """
        Samples uniformly on state space hypercube
        Returns 1 sample
        """
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        # https://math.stackexchange.com/questions/2687807/uniquely-identify-hypercube-faces
        which_facet_pair = np.random.choice(np.arange(self.x_dim), p=self.vols)
        which_facet = np.random.choice([0, 1])

        # samples = np.random.uniform(low=self.x_lim[:, 0], high=self.x_lim[:, 1])
        unif = torch.rand(self.x_dim).to(self.device)
        sample = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]
        sample[which_facet_pair] = self.x_lim[which_facet_pair, which_facet]
        return sample

    def _sample_in_safe_set(self, phi_fn, random_seed=None):

        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        bs = 100 # TODO: assuming this can use GPU
        N_samp = 1 # 1 sample desired
        N_samp_found = 0
        i = 0
        while N_samp_found < N_samp:
            # Sample in box
            unif = torch.rand((bs, self.x_dim)).to(self.device)
            samples_torch = unif*(self.x_lim[:, 1] - self.x_lim[:, 0]) + self.x_lim[:, 0]

            # Check if samples in invariant set
            phi_vals = phi_fn(samples_torch)
            max_phi_vals = torch.max(phi_vals, dim=1)[0]

            # Save good samples
            ind = torch.argwhere(max_phi_vals <= 0).flatten()
            samples_torch_inside = samples_torch[ind]
            N_samp_found += len(ind)
            i += 1

        # Could be more than N_samp currently; truncate to exactly N_samp
        rv = samples_torch_inside[0] # is flat shape already
        return rv

    def _sample_in_gaussian(self, safe_set_sample):
        cov = 2*self.gaussian_t*torch.eye(self.x_dim).to(self.device)
        m = torch.distributions.MultivariateNormal(safe_set_sample, cov)
        sample_torch = m.sample()
        return sample_torch

    def _intersect_segment_with_manifold(self, p1, p2, phi_fn):
        diff = p2-p1

        left_weight = 0.0
        right_weight = 1.0
        # left_val = surface_fn(p1.view(1, -1)).item()
        # right_val = surface_fn(p2.view(1, -1)).item()
        left_val = phi_fn(p1.view(1, -1))[:, -1].item()
        right_val = phi_fn(p2.view(1, -1))[:, -1].item()
        left_sign = np.sign(left_val)
        right_sign = np.sign(right_val)

        if left_sign*right_sign > 0:
            # print("does not intersect")
            return None

        t0 = time.perf_counter()
        while True:
            # print(left_weight, right_weight)
            # print(left_val, right_val)
            mid_weight = (left_weight + right_weight)/2.0
            mid_point = p1 + mid_weight*diff

            # mid_val = surface_fn(mid_point.view(1, -1)).item()
            mid_val = phi_fn(mid_point.view(1, -1))[:, -1].item()
            mid_sign = np.sign(mid_val)
            if mid_sign*left_sign < 0:
                # go to the left side
                right_weight = mid_weight
                right_val = mid_val
            elif mid_sign*right_sign <= 0:
                left_weight = mid_weight
                left_val = mid_val

            if max(abs(left_val), abs(right_val)) < self.projection_tolerance:
                intersection_point = p1 + left_weight*diff
                break
            t1 = time.perf_counter()
            if (t1-t0)>7: # an arbitrary time limit
                print("Something is wrong in projection")
                print(left_val, right_val)
                print(left_weight, right_weight)
                print("p1:", p1)
                print("p2:", p2)
                return None
        # print("success")
        return intersection_point

    def _sample_segment_intersect_boundary(self, phi_fn, random_seed=None):
        # boundary_sampling_method; ["uniform", "gaussian"]
        if self.boundary_sampling_method == "uniform":
            outer = self._sample_on_cube(random_seed=random_seed)
            center = self._sample_in_cube(random_seed=random_seed)
        elif self.boundary_sampling_method == "gaussian":
            center = self._sample_in_safe_set(phi_fn, random_seed=random_seed)
            outer = self._sample_in_gaussian(center)

        intersection = self._intersect_segment_with_manifold(center, outer, phi_fn)
        return intersection

    def _sample_points_on_boundary_sequential(self, phi_fn, n_samples):
        """
        Returns torch array of size (self.n_samples, self.x_dim)
        """
        t0 = time.perf_counter()
        # Everything done in torch
        samples = torch.zeros((0, self.x_dim)).to(self.device)
        n_remaining_to_sample = n_samples

        n_segments_sampled = 0
        while n_remaining_to_sample > 0:
            if self.verbose:
                print(".", end=" ")
            intersection = self._sample_segment_intersect_boundary(phi_fn)
            if intersection is not None:
                # samples.append(intersection.view(1, -1))
                samples = torch.cat((samples, intersection.view(1, -1)), dim=0)
                n_remaining_to_sample -= 1
                if self.verbose:
                    print("\n")
                    print(n_remaining_to_sample)

            n_segments_sampled += 1
            # self.logger.info("%i segments" % n_segments_sampled)

        # samples = torch.cat(samples, dim=0)
        # self.logger.info("Done with sampling points on the boundary...")
        tf = time.perf_counter()
        debug_dict = {"t_sample_boundary": (tf- t0), "n_segments_sampled": n_segments_sampled}
        return samples, debug_dict

    def _sample_points_on_boundary_gpu_parallelized(self, phi_fn, n_samples):

        t0 = time.perf_counter()
        phi_fn.share_memory() # Adds to Queue, which is shared between processes?

        # Everything done in torch
        samples = []
        n_remaining_to_sample = n_samples

        random_start = 0
        random_bs = 1000
        random_seeds = np.arange(random_start, random_start + random_bs * n_samples) # This should be enough. If it isn't, code will error out
        np.random.shuffle(random_seeds)  # in place

        it = 0
        while n_remaining_to_sample > 0:
            batch_random_seeds = random_seeds[it * self.n_gpu:(it + 1) * self.n_gpu]
            if batch_random_seeds.size < self.n_gpu:
                # need more random seeds
                random_start = random_start + random_bs * n_samples
                random_seeds = np.arange(random_start,
                                         random_start + random_bs * n_samples)  # This should be enough. If it isn't, code will error out
                np.random.shuffle(random_seeds)

            final_arg = [[phi_fn, batch_random_seeds[i]] for i in range(self.n_gpu)]
            result = self.pool.starmap(self._sample_segment_intersect_boundary, final_arg)

            for intersection in result:
                if intersection is not None:
                    samples.append(intersection.view(1, -1))
                    n_remaining_to_sample -= 1 # could go negative!

            it += 1
            # self.logger.info("%i segments sampled" % (it*self.n_gpu))


        samples = torch.cat(samples[:n_samples], dim=0)
        # self.logger.info("Done with sampling points on the boundary...")
        tf = time.perf_counter()
        debug_dict = {"t_sample_boundary": (tf- t0), "n_segments_sampled": (it*self.n_gpu)}
        return samples, debug_dict

    def _sample_points_on_boundary(self, phi_fn, n_samples):
        """
        Returns torch array of size (self.n_samples, self.x_dim)
        # boundary_sampling_option: ["sequential", "gpu_parallelized", "cpu_parallelized"]
        # boundary_sampling_method; ["uniform", "gaussian"]
        """
        if self.boundary_sampling_speedup_method == "gpu_parallelized":
            samples, debug_dict = self._sample_points_on_boundary_gpu_parallelized(phi_fn, n_samples)
        elif self.boundary_sampling_speedup_method == "sequential":
            samples, debug_dict = self._sample_points_on_boundary_sequential(phi_fn, n_samples)
        elif self.boundary_sampling_speedup_method == "cpu_parallelized":
            print("self.boundary_sampling_option == cpu_parallelized hasn't been implemented....")
            raise NotImplementedError
        return samples, debug_dict

    def opt(self, objective_fn, phi_fn, iteration, debug=False):
        t0_opt = time.perf_counter()

        if self.X_saved is None:
            X_init, boundary_sample_debug_dict = self._sample_points_on_boundary(phi_fn, self.n_samples)

            X_reuse_init = torch.zeros((0, self.x_dim))
            X_random_init = X_init
        else:
            # print("inside attacker, using saved points")
            # print("check that X+saved format correct; code compiles")
            # IPython.embed()
            n_target_reuse_samples = int(self.n_samples*self.p_reuse)

            inds = torch.argsort(self.obj_vals_saved, axis=0, descending=True).flatten()

            # Some attacks will be very near each other. This helps to only select distinct attacks
            inds_distinct = [inds[0]]
            for ind in inds[1:]:
                diff = self.X_saved[torch.tensor(inds_distinct)] - self.X_saved[ind]
                distances = torch.norm(diff.view(-1, self.x_dim), dim=1)
                if torch.any(distances <= 1e-1).item(): # TODO: set this (distance which determines an "identical" point)
                    print("passed")
                    continue
                inds_distinct.append(ind)
                if len(inds_distinct) >= n_target_reuse_samples:
                    break

            # IPython.embed()
            n_reuse_samples = len(inds_distinct)
            n_random_samples= self.n_samples - n_reuse_samples
            # print("Actual percentage reuse: %f" % ((n_reuse_samples/self.n_samples)*100))
            X_reuse_init = self.X_saved[torch.tensor(inds_distinct)]
            # print("Reprojecting")
            X_reuse_init = self._project(phi_fn, X_reuse_init) # reproject, since phi changed
            # print("Sampling points on boundary")
            X_random_init, boundary_sample_debug_dict = self._sample_points_on_boundary(phi_fn, n_random_samples)
            # print("Done")
            X_init = torch.cat([X_random_init, X_reuse_init], axis=0)

        tf_init = time.perf_counter()

        X = X_init.clone()
        i = 0
        early_stopping = EarlyStoppingBatch(self.n_samples, patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)
        # logging
        t_grad_step = []
        t_reproject = []
        dist_diff_after_proj = []
        obj_vals = objective_fn(X.view(-1, self.x_dim))
        init_best_attack_value = torch.max(obj_vals).item()

        # train_attacker_use_n_step_schedule
        max_n_steps = self.max_n_steps
        if self.train_attacker_use_n_step_schedule:
            max_n_steps = (0.5*self.max_n_steps)*np.exp(-iteration/75) + self.max_n_steps
            print("Max_n_steps: %i" % max_n_steps)
        while True:
            if self.verbose:
                print("Counterex. max. step #%i" % i)
            # IPython.embed()
            X, step_debug_dict = self._step(objective_fn, phi_fn, X) # Take gradient steps on all candidate attacks
            # obj_vals = objective_fn(X.view(-1, self.x_dim))

            # Logging
            t_grad_step.append(step_debug_dict["t_grad_step"])
            t_reproject.append(step_debug_dict["t_reproject"])
            dist_diff_after_proj.append(step_debug_dict["dist_diff_after_proj"])

            # Loop break condition
            if self.stopping_condition == "n_steps":
                if (i > max_n_steps):
                    break
            elif self.stopping_condition == "early_stopping":
                print("Not recommended to use this option; it will run for hundreds of steps before stopping")
                raise NotImplementedError
            i += 1

        tf_opt = time.perf_counter()

        # Save for warmstart
        self.X_saved = X
        obj_vals = objective_fn(X.view(-1, self.x_dim))
        self.obj_vals_saved = obj_vals

        # Returning a single attack
        max_ind = torch.argmax(obj_vals)

        if not debug:
            x = X[max_ind]
            return x, {}
        else:
            x = X[max_ind]
            final_best_attack_value = torch.max(obj_vals).item()

            t_init = tf_init - t0_opt
            t_total_opt = tf_opt - t0_opt

            # TODO: do not change the names in the dict here! Names are matched to trainer.py
            debug_dict = {"X_init": X_init, "X_init_reuse": X_reuse_init, "X_init_random": X_random_init, "X_final": X, "X_obj_vals": obj_vals, "init_best_attack_value": init_best_attack_value, "final_best_attack_value": final_best_attack_value, "t_init": t_init, "t_grad_steps": t_grad_step, "t_reproject": t_reproject, "t_total_opt": t_total_opt, "dist_diff_after_proj": dist_diff_after_proj, "n_opt_steps": max_n_steps}
            debug_dict.update(boundary_sample_debug_dict) #  {"t_sample_boundary": (tf- t0), "n_segments_sampled": n_segments_sampled}

            # print("check before returning from opt")
            # IPython.embed()
            return x, debug_dict
