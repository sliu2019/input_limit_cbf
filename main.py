import torch
from torch import nn
from torch.autograd import grad

# from src.attacks.basic_attacker import BasicAttacker
from src.attacks.batch_warmstart_attacker import BatchWarmstartAttacker

from src.trainer import Trainer

from src.reg_samplers.boundary import BoundaryRegSampler
from src.reg_samplers.random import RandomRegSampler
from src.reg_samplers.fixed import FixedRegSampler
from src.reg_samplers.random_inside import RandomInsideRegSampler
reg_samplers_name_to_class_dict = {"boundary": BoundaryRegSampler, "random": RandomRegSampler, "fixed": FixedRegSampler, "random_inside": RandomInsideRegSampler}

# from src.cbf_designs.baseline_cbf import BaselineCBF
from src.cbf_designs.neural_cbf import NeuralCBF

from src.utils import *
from src.argument import create_parser, print_args

import os
import math
import pickle

# TODO: comment this out before a run
# from global_settings import *

class Objective(nn.Module):
	def __init__(self, phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

	def forward(self, x):
		# The way these are implemented should be batch compliant
		u_lim_set_vertices = self.uvertices_fn(x) # (bs, n_vertices, u_dim), can be a function of x_batch
		n_vertices = u_lim_set_vertices.size()[1]

		# Evaluate every X against multiple U
		U = torch.reshape(u_lim_set_vertices, (-1, self.u_dim)) # (bs x n_vertices, u_dim)
		X = (x.unsqueeze(1)).repeat(1, n_vertices, 1) # (bs, n_vertices, x_dim)
		X = torch.reshape(X, (-1, self.x_dim)) # (bs x n_vertices, x_dim)

		xdot = self.xdot_fn(X, U)

		orig_req_grad_setting = x.requires_grad
		x.requires_grad = True
		phi_value = self.phi_fn(x)
		grad_phi = grad([torch.sum(phi_value[:, -1])], x, create_graph=True)[0] # check
		x.requires_grad = orig_req_grad_setting

		grad_phi = (grad_phi.unsqueeze(1)).repeat(1, n_vertices, 1)
		grad_phi = torch.reshape(grad_phi, (-1, self.x_dim))

		# Dot product
		phidot_cand = xdot.unsqueeze(1).bmm(grad_phi.unsqueeze(2))
		phidot_cand = torch.reshape(phidot_cand, (-1, n_vertices)) # bs x n_vertices

		phidot, _ = torch.min(phidot_cand, 1)

		# if self.args.no_softplus_on_obj:
		# 	result = phidot
		# else:
		# 	result = nn.functional.softplus(phidot) # using softplus on loss!!!
		result = phidot
		result = result.view(-1, 1) # ensures bs x 1

		return result

class Regularizer(nn.Module):
	def __init__(self, phi_fn, device, reg_weight=0.0, reg_transform="sigmoid"):
		super().__init__()
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		assert reg_weight >= 0.0

	def forward(self, x):
		all_phi_values = self.phi_fn(x)
		max_phi_values = torch.max(all_phi_values, dim=1)[0]

		if self.reg_transform == "sigmoid":
			transform_of_max_phi = nn.functional.sigmoid(0.3*max_phi_values)
		elif self.reg_transform == "softplus":
			transform_of_max_phi = nn.functional.softplus(max_phi_values)
		reg = self.reg_weight*torch.mean(transform_of_max_phi)
		return reg

def create_flying_param_dict(args=None):
	# Args: for modifying the defaults through args
	param_dict = {
		"m": 0.8,
		"J_x": 0.005,
		"J_y": 0.005,
		"J_z": 0.009,
		"l": 1.5,
		"k1": 4.0,
		"k2": 0.05,
		"m_p": 0.04, # 5% of quad weight
		"L_p": 3.0, # Prev: 0.03
		'delta_safety_limit': math.pi / 4  # should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	state_index_names = ["gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha", "phi", "theta", "dphi",
	                     "dtheta"]  # excluded x, y, z
	state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))

	r = 2
	x_dim = len(state_index_names)
	u_dim = 4
	ub = args.box_ang_vel_limit
	thresh = np.array([math.pi / 3, math.pi / 3, math.pi, ub, ub, ub, math.pi / 3, math.pi / 3, ub, ub],
	                  dtype=np.float32) # angular velocities bounds probably much higher in reality (~10-20 for drone, which can do 3 flips in 1 sec).

	x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)  # (13, 2)

	# Save stuff in param dict
	param_dict["state_index_dict"] = state_index_dict
	param_dict["r"] = r
	param_dict["x_dim"] = x_dim
	param_dict["u_dim"] = u_dim
	param_dict["x_lim"] = x_lim

	# write args into the param_dict
	param_dict["L_p"] = args.pend_length

	return param_dict

def main(args):
	# Boilerplate for saving
	save_folder = '%s_%s' % (args.problem, args.affix)

	log_folder = os.path.join(args.log_root, save_folder)
	model_folder = os.path.join(args.model_root, save_folder)

	makedirs(log_folder)
	makedirs(model_folder)

	setattr(args, 'log_folder', log_folder)
	setattr(args, 'model_folder', model_folder)

	logger = create_logger(log_folder, 'train', 'info')
	print_args(args, logger)

	args_savepth = os.path.join(log_folder, "args.txt")
	save_args(args, args_savepth)

	# Device
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
		dev = "cuda:%i" % (args.gpu)
		# print("Using GPU device: %s" % dev)
	else:
		dev = "cpu"
	device = torch.device(dev)

	# Selecting problem
	if args.problem == "quadcopter_pend":
		param_dict = create_flying_param_dict(args)

		r = param_dict["r"]
		x_dim = param_dict["x_dim"]
		u_dim = param_dict["u_dim"]
		x_lim = param_dict["x_lim"]

		# Create phi
		from src.problems.quadcopter_pend import RhoMax, RhoSum, XDot, ULimitSetVertices
		if args.rho == "sum":
			h_fn = RhoSum(param_dict)
		elif args.rho == "max":
			h_fn = RhoMax(param_dict)

		xdot_fn = XDot(param_dict, device)
		uvertices_fn = ULimitSetVertices(param_dict, device)

		reg_sampler = reg_samplers_name_to_class_dict[args.reg_sampler](x_lim, device, logger, n_samples=args.reg_n_samples)

		if args.phi_include_xe:
			x_e = torch.zeros(1, x_dim)
		else:
			x_e = None

		# Passing in subset of state to NN
		from src.utils import IndexNNInput, TransformEucNNInput
		state_index_dict = param_dict["state_index_dict"]
		if args.phi_nn_inputs == "spherical":
			nn_input_modifier = None
		elif args.phi_nn_inputs == "euc":
			nn_input_modifier = TransformEucNNInput(state_index_dict)
	else:
		raise NotImplementedError

	# Save param_dict
	with open(os.path.join(log_folder, "param_dict.pkl"), 'wb') as handle:
		pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Send all modules to the correct device
	h_fn = h_fn.to(device)
	xdot_fn = xdot_fn.to(device)
	uvertices_fn = uvertices_fn.to(device)
	if x_e is not None:
		x_e = x_e.to(device)
	x_lim = torch.tensor(x_lim).to(device)

	# Create CBF, etc.
	phi_fn = NeuralCBF(h_fn, xdot_fn, r, x_dim, u_dim, device, args, x_e=x_e, nn_input_modifier=nn_input_modifier)
	objective_fn = Objective(phi_fn, xdot_fn, uvertices_fn, x_dim, u_dim, device, logger, args)
	reg_fn = Regularizer(phi_fn, device, reg_weight=args.reg_weight, reg_transform=args.reg_transform)

	# Send remaining modules to the correct device
	phi_fn = phi_fn.to(device)
	objective_fn = objective_fn.to(device)
	reg_fn = reg_fn.to(device)

	# Create attacker
	if args.train_attacker == "batch_warmstart":
		attacker = BatchWarmstartAttacker(x_lim, device, logger, n_samples=args.train_attacker_n_samples,
		                                  stopping_condition=args.train_attacker_stopping_condition,
		                                  max_n_steps=args.train_attacker_max_n_steps,
		                                  lr=args.train_attacker_lr,
		                                  projection_tolerance=args.train_attacker_projection_tolerance,
		                                  projection_lr=args.train_attacker_projection_lr,
		                                  projection_time_limit=args.train_attacker_projection_time_limit,
		                                  train_attacker_use_n_step_schedule=args.train_attacker_use_n_step_schedule,
		                                  boundary_sampling_speedup_method=args.batch_warmstart_speedup_method, boundary_sampling_method=args.batch_warmstart_sampling_method,
		                                  gaussian_t=args.batch_warmstart_gaussian_t,
		                                  p_reuse=args.train_attacker_p_reuse)

	# Create test attacker
	# Note: doesn't matter that we're passing train params. We're only using test_attacker to sample on boundary
	test_attacker = BatchWarmstartAttacker(x_lim, device, logger, n_samples=args.train_attacker_n_samples,
	                                       stopping_condition=args.train_attacker_stopping_condition,
	                                       max_n_steps=args.train_attacker_max_n_steps,
	                                       lr=args.train_attacker_lr,
	                                       projection_tolerance=args.train_attacker_projection_tolerance,
	                                       projection_lr=args.train_attacker_projection_lr,
	                                       projection_time_limit=args.train_attacker_projection_time_limit,
	                                       train_attacker_use_n_step_schedule=args.train_attacker_use_n_step_schedule,
	                                       boundary_sampling_speedup_method=args.batch_warmstart_speedup_method,
	                                       boundary_sampling_method=args.batch_warmstart_sampling_method,
	                                       gaussian_t=args.batch_warmstart_gaussian_t,
	                                       p_reuse=args.train_attacker_p_reuse)

	# Pass everything to Trainer
	trainer = Trainer(args, logger, attacker, test_attacker, reg_sampler, param_dict, device)
	trainer.train(objective_fn, reg_fn, phi_fn, xdot_fn)


if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_known_args()[0]
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)
	main(args)
