import argparse
import math

def create_parser():
	# Problem
	parser = argparse.ArgumentParser(description='CBF synthesis')
	parser.add_argument('--problem', default='quadcopter_pend', help='problem specifies dynamics, h definition, U_limits, etc.', choices=["quadcopter_pend"])

	# rho(x) (user-specified SI)
	parser.add_argument('--rho', type=str, default='sum', choices=['max', 'sum'], help='For quadcopter_pend, chose the form of rho(x)')

	# Phi
	parser.add_argument('--phi_nn_dimension', default="64-64", type=str, help='for neural CBF: specify the hidden dimension')
	parser.add_argument('--phi_nnl', default="tanh-tanh-none", type=str, help='for neural CBF: can also do tanh-tanh-softplus')
	parser.add_argument('--phi_ci_init_range', default=1e-2, type=float, help='for neural CBF: c_i are initialized uniformly within the range [0, x]')
	parser.add_argument('--phi_include_xe', action='store_true', help='for neural CBF')
	parser.add_argument('--phi_nn_inputs', type=str, default="spherical", choices=["spherical", "euc"], help='for neural CBF: which coordinates? spherical or euclidean')

	# Parameters for quadcopter-pend only
	parser.add_argument('--pend_length', default=3.0, type=float)
	parser.add_argument('--box_ang_vel_limit', default=20.0, type=float)

	# Reg
	parser.add_argument('--reg_weight', default=0.0, type=float, help='the weight on the volume term')
	parser.add_argument('--reg_sampler', type=str, default="random", choices=['boundary', 'random', 'fixed', 'random_inside', 'random_inside_then_boundary'], help="random_inside_then_boundary switches from RI to bdry after vol drops")
	parser.add_argument('--reg_n_samples', type=int, default=250)
	parser.add_argument('--reg_transform', type=str, default="sigmoid", choices=["sigmoid", "softplus"])

	# Objective
	parser.add_argument('--objective_option', type=str, default='regular', choices=['regular', 'softplus', 'weighted_average', 'weighted_average_include_neg_phidot'], help="allow negative pays attention to phi < 0 as well")

	# Attacker: train
	parser.add_argument('--train_attacker', default='batch_warmstart', choices=['batch_warmstart'])
	parser.add_argument("--batch_warmstart_speedup_method", type=str, default="sequential", choices=["sequential", "gpu_parallelized", "cpu_parallelized"])
	parser.add_argument("--batch_warmstart_sampling_method", type=str, default="uniform", choices=["uniform", "gaussian"])
	parser.add_argument("--batch_warmstart_gaussian_t", type=float, default=1.0)

	# Gradient batch attacker
	parser.add_argument('--train_attacker_n_samples', default=60, type=int)
	parser.add_argument('--train_attacker_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	parser.add_argument('--train_attacker_max_n_steps', default=50, type=int)
	parser.add_argument('--train_attacker_p_reuse', default=0.7, type=float)
	parser.add_argument('--train_attacker_projection_tolerance', default=1e-1, type=float, help='when to consider a point "projected"')
	parser.add_argument('--train_attacker_projection_lr', default=1e-2, type=float) # changed from 1e-4 to increase proj speed
	parser.add_argument('--train_attacker_projection_time_limit', default=3.0, type=float)
	parser.add_argument('--train_attacker_lr', default=1e-3, type=float)

	parser.add_argument('--train_attacker_use_n_step_schedule', action='store_true', help='use a schedule (starting with >>>max_n_steps and exponentially decreasing down to it')

	# Attacker: test
	parser.add_argument('--test_N_volume_samples', default=2500, type=int)
	parser.add_argument('--test_N_boundary_samples', default=2500, type=int)

	# Trainer
	parser.add_argument('--trainer_stopping_condition', default='n_steps', choices=['n_steps', 'early_stopping'])
	parser.add_argument('--trainer_early_stopping_patience', default=100, type=int)
	parser.add_argument('--trainer_n_steps', default=3000, type=int, help='if stopping condition is n_steps, specify the number here')
	parser.add_argument('--trainer_lr', default=1e-3, type=float)

	# Saving/logging
	parser.add_argument('--random_seed', default=1, type=int)
	parser.add_argument('--affix', default='default', help='the affix for the save folder')
	parser.add_argument('--log_root', default='log',
	                    help='the directory to save the logs or other imformations (e.g. images)')
	parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
	parser.add_argument('--n_checkpoint_step', type=int, default=5,
	                    help='number of iterations to save a checkpoint')
	parser.add_argument('--n_test_loss_step', type=int, default=25,
	                    help='number of iterations to compute test loss; if negative, then never')

	# Misc
	parser.add_argument('--gpu', '-g', default=0, type=int, help='which gpu to use')
	return parser

def print_args(args, logger=None):
	for k, v in vars(args).items():
		if logger is not None:
			logger.info('{:<16} : {}'.format(k, v))
		else:
			print('{:<16} : {}'.format(k, v))