import torch
from torch import nn
import torch.optim as optim

from src.utils import save_model
import os
import time
import IPython
from torch.autograd import grad
from src.utils import *
import datetime
import pickle
import math

class Trainer():
	def __init__(self, args, logger, attacker, test_attacker, reg_sampler, param_dict, device):
		vars = locals()  # dict of local names
		self.__dict__.update(vars)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`

		self.save_folder = '%s_%s' % (self.args.problem, self.args.affix)
		self.log_folder = os.path.join(self.args.log_root, self.save_folder)
		self.data_save_fpth = os.path.join(self.log_folder, "data.pkl")

		# For approximating volume
		x_lim = param_dict["x_lim"]
		self.x_lim = x_lim
		self.x_dim = x_lim.shape[0]
		self.x_lim_interval_sizes = np.reshape(x_lim[:, 1] - x_lim[:, 0], (1, self.x_dim))

	def train(self, objective_fn, reg_fn, phi_fn, xdot_fn):
		##################################
		######### Set up saving ##########
		##################################

		data_dict = {
			"train_loop_times": [],
			"train_losses": [],
			"train_attack_losses": [],
			"train_reg_losses": [],
			"grad_norms": [],
			"V_approx_list": [],
			"boundary_samples_obj_values": [],
			"test_t_total": [],
			"test_t_boundary": []
		}

		data_dict["ci_list"] = []
		data_dict["k0_list"] = []

		train_attack_dict = {"train_attacks": [],
			"train_attack_X_init": [],
			"train_attack_X_init_reuse": [],
			"train_attack_X_init_random": [],
			"train_attack_X_final": [],
			"train_attack_X_obj_vals": [],
			"train_attack_X_phi_vals": [],
			"train_attack_init_best_attack_value": [],
			"train_attack_final_best_attack_value": [],
			"train_attack_t_init": [],
			"train_attack_t_grad_steps": [],
			"train_attack_t_reproject": [],
			"train_attack_t_total_opt": [],
			"train_attack_t_sample_boundary": [],
		    "train_attack_n_segments_sampled": [],
		    "train_attack_dist_diff_after_proj": [],
		    "train_attack_n_opt_steps": []
		    }

		data_dict.update(train_attack_dict)

		reg_debug_dict = {"reg_grad_norms": []}

		data_dict.update(reg_debug_dict)

		###########  Done  ###########
		##############################
		p_dict = {p[0]:p[1] for p in phi_fn.named_parameters()}
		pos_params = [p_dict[name] for name in phi_fn.pos_param_names]

		optimizer = optim.Adam(phi_fn.parameters(), lr=self.args.trainer_lr)

		early_stopping = EarlyStopping(patience=self.args.trainer_early_stopping_patience, min_delta=1e-2)

		_iter = 0
		t0 = time.perf_counter()

		file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
		save_model(phi_fn, file_name)

		# print("Before training")
		# IPython.embed()
		while True:

			iteration_info_dict = {}
			X_reg = self.reg_sampler.get_samples(phi_fn)
			reg_value = reg_fn(X_reg)

			# Note: this won't work for the regular attacker
			x, debug_dict = self.attacker.opt(objective_fn, phi_fn, _iter, debug=True)
			X = debug_dict["X_final"]

			if self.args.objective_option == "regular":
				x_batch = x.view(1, -1)
				attack_value = objective_fn(x_batch)[0, 0]
			elif self.args.objective_option == "softplus":
				x_batch = x.view(1, -1)
				attack_value = nn.functional.softplus(objective_fn(x_batch)[0, 0])
			elif self.args.objective_option == "weighted_average":
				c = 0.1
				obj = objective_fn(X)
				pos_inds = torch.where(obj >= 0) # tuple of 2D inds
				pos_obj = c*obj[pos_inds[0], pos_inds[1]]
				with torch.no_grad():
					w = torch.nn.functional.softmax(pos_obj)
				attack_value = torch.dot(w.flatten(), pos_obj.flatten())
			elif self.args.objective_option == "weighted_average_include_neg_phidot":
				# Eliminates the "relu" effect on above
				c = 0.1

				obj = objective_fn(X)
				w = torch.exp(c*obj)
				w = w/torch.sum(w)
				attack_value = torch.dot(w.flatten(), obj.flatten())

			# For logging
			x_batch = x.view(1, -1)
			max_value = objective_fn(x_batch)[0, 0]

			objective_value = attack_value + reg_value

			#######################################################
			############# Now, taking the gradients ###############
			#######################################################
			optimizer.zero_grad()

			reg_value.backward()

			##### Check reg gradient #####
			avg_grad_norm = 0
			n_param = 0
			for n, p in phi_fn.named_parameters():
				if n not in phi_fn.exclude_from_gradient_param_names:
				# if n not in ["ci", "k0"]:
					avg_grad_norm += torch.linalg.norm(p.grad).item()
					n_param += 1
			avg_grad = avg_grad_norm/n_param
			iteration_info_dict["reg_grad_norms"] = avg_grad
			self.logger.info(f'Reg grad norm: {avg_grad:.3f}')
			#*****************************

			attack_value.backward()

			##### Check total gradient #####
			avg_grad_norm = 0
			n_param = 0
			for n, p in phi_fn.named_parameters():
				# if n not in ["ci", "k0"]:
				if n not in phi_fn.exclude_from_gradient_param_names:
					avg_grad_norm += torch.linalg.norm(p.grad).item()
					n_param += 1
			avg_grad = avg_grad_norm/n_param
			iteration_info_dict["grad_norms"] = avg_grad
			self.logger.info(f'total grad norm: {avg_grad:.3f}')
			#*****************************

			optimizer.step()

			with torch.no_grad():
				for param in pos_params:
					pos_param = torch.maximum(param, torch.zeros_like(param))
					param.copy_(pos_param)

			tnow = time.perf_counter()

			#######################################################
			############## Logging and appending data #############
			#######################################################
			# TODO: Make sure you detach before logging, otherwise you will accumulate memory over iterations and get an OOM
			self.logger.info('\n' + '=' * 20 + f' evaluation at iteration: {_iter} ' \
			                 + '=' * 20)

			self.logger.info(f'train total loss: {objective_value:.3f}%')
			self.logger.info(f'train max loss: {max_value:.3f}%, reg loss: {reg_value:.3f}%')
			t_so_far = tnow-t0
			t_so_far_str = str(datetime.timedelta(seconds=t_so_far))

			# Timing logging + saving
			t_init = debug_dict["t_init"]
			t_grad_steps = debug_dict["t_grad_steps"]
			t_reproject = debug_dict["t_reproject"]
			t_total_opt = debug_dict["t_total_opt"]

			self.logger.info('time spent training so far: %s' % t_so_far_str)
			self.logger.info(f'train attack total time: {t_total_opt:.3f}s')
			self.logger.info(f'train attack init time: {t_init:.3f}s')
			self.logger.info(f'train attack avg grad step time: {np.mean(t_grad_steps):.3f}s')
			self.logger.info(f'train attack avg reproj time: {np.mean(t_reproject):.3f}s')

			iteration_info_dict["train_loop_times"] = t_so_far

			# debug logging
			self.logger.info("\n")
			self.logger.info(f'train attack loss increase over inner max: {(debug_dict["final_best_attack_value"]-debug_dict["init_best_attack_value"]):.3f}')
			# mem leak logging
			self.logger.info('OOM debug. Mem allocated and reserved: %f, %f' % (torch.cuda.memory_allocated(self.args.gpu), torch.cuda.memory_reserved(self.args.gpu)))

			# Losses saving
			iteration_info_dict["train_attack_losses"] = max_value # TODO
			iteration_info_dict["train_reg_losses"] = reg_value
			iteration_info_dict["train_losses"] = objective_value

			# Train attack saving
			iteration_info_dict["train_attacks"] = x
			iteration_info_dict["train_attack_X_phi_vals"] = phi_fn(X)
			debug_dict = {"train_attack_" + key: value for key, value in debug_dict.items()}

			iteration_info_dict.update(debug_dict)

			# Misc saving
			iteration_info_dict["ci_list"] = phi_fn.ci
			iteration_info_dict["k0_list"] = phi_fn.k0
			print(phi_fn.k0)
			print(phi_fn.ci)

			# Merge into info_dict
			for key, value in iteration_info_dict.items():
				if torch.is_tensor(value):
					value = value.detach().cpu().numpy()
				data_dict[key].append(value)

			# Rest of print output
			self.logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')
			#######################################################
			##############    Saving data   #######################
			#######################################################
			if _iter % self.args.n_checkpoint_step == 0:
				file_name = os.path.join(self.args.model_folder, f'checkpoint_{_iter}.pth')
				save_model(phi_fn, file_name)

				##############################
				######### Save data ##########
				##############################
				print("Saving at: ", self.data_save_fpth)
				with open(self.data_save_fpth, 'wb') as handle:
					pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

			# print("After first training iteration")
			# IPython.embed()
			#######################################################
			##############   Compute test stats   #################
			#######################################################
			# TODO: the fact that this is not = self.args.n_checkpoint_step necessarily means that you might have to refactor stuff in flying_rollout_experiment
			if _iter % self.args.n_test_loss_step == 0:
				t0_test = time.perf_counter()

				samp_numpy = np.random.uniform(size=(self.args.test_N_volume_samples, self.x_dim)) * self.x_lim_interval_sizes + self.x_lim[:, [0]].T
				samp_torch = torch.from_numpy(samp_numpy.astype("float32")).to(self.device)
				M = 100

				N_samples_inside = 0
				for k in range(math.ceil(self.args.test_N_volume_samples/float(M))):
					phi_vals_batch = phi_fn(samp_torch[k*M: min((k+1)*M, self.args.test_N_volume_samples)])
					N_samples_inside += torch.sum(torch.max(phi_vals_batch, axis=1)[0] <= 0.0)
				V_approx = N_samples_inside*100.0/float(self.args.test_N_volume_samples)
				V_approx = V_approx.item()
				data_dict["V_approx_list"].append(V_approx)

				# Sample on boundary
				t0_test_boundary = time.perf_counter()
				boundary_samples, debug_dict = self.test_attacker._sample_points_on_boundary(phi_fn, self.args.test_N_boundary_samples) # test_attacker now using "faster" version
				boundary_samples_obj_value = objective_fn(boundary_samples)
				boundary_samples_obj_value = boundary_samples_obj_value.detach().cpu().numpy()
				data_dict["boundary_samples_obj_values"].append(boundary_samples_obj_value)

				self.logger.info('\n' + '+' * 20 + f' computing test stats ' \
				                 + '+' * 20)
				self.logger.info(f'v approx: {V_approx:.3f}% of volume')
				percent_infeas_at_boundary = np.sum(boundary_samples_obj_value > 0)*100/boundary_samples_obj_value.size
				self.logger.info(f'percentage infeasible at boundary: {percent_infeas_at_boundary:.2f}%')
				infeas_values = (boundary_samples_obj_value > 0)*boundary_samples_obj_value
				average_infeas_amount = np.mean(infeas_values)
				std_infeas_amount = np.std(infeas_values)
				self.logger.info(f'mean, std amount infeasible at boundary: {average_infeas_amount:.2f} +/- {std_infeas_amount:.2f}')
				self.logger.info(f'max amount infeasible at boundary: {np.max(infeas_values):.2f}')
				self.logger.info('\n' + '+' * 80)

				tf_test = time.perf_counter()

				data_dict["test_t_total"].append(tf_test-t0_test)
				data_dict["test_t_boundary"].append(tf_test-t0_test_boundary)

			if self.args.trainer_stopping_condition == "early_stopping":
				# early_stopping(test_loss) # TODO: should technically use test_loss, but we're not computing it in the loop
				early_stopping(objective_value)
				if early_stopping.early_stop:
					break
			elif self.args.trainer_stopping_condition == "n_steps":
				if _iter > self.args.trainer_n_steps:
					break
			_iter += 1

