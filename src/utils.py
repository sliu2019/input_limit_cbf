import os
import json
import logging

import numpy as np
import torch
from torch import nn
import IPython
from dotmap import DotMap
import torch
import pickle
from src.argument import create_parser, print_args

def create_logger(save_path='', file_type='', level='debug'):

	if level == 'debug':
		_level = logging.DEBUG
	elif level == 'info':
		_level = logging.INFO

	logger = logging.getLogger()
	logger.setLevel(_level)

	cs = logging.StreamHandler()
	cs.setLevel(_level)
	logger.addHandler(cs)

	if save_path != '':
		file_name = os.path.join(save_path, file_type + '_log.txt')
		fh = logging.FileHandler(file_name, mode='w')
		fh.setLevel(_level)

		logger.addHandler(fh)

	return logger


def save_model(model, file_name):
	torch.save(model.state_dict(), file_name)

def load_model(model, file_name):
	model.load_state_dict(
		torch.load(file_name, map_location=lambda storage, loc: storage))

def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

def save_args(args, file_name):
	with open(file_name, 'w') as f:
		json.dump(args.__dict__, f, indent=2)

def load_args(file_name):
	parser = create_parser()
	args = parser.parse_known_args()[0]
	# args = parser() # TODO
	with open(file_name, 'r') as f:
		args.__dict__ = json.load(f)
	return args


class EarlyStopping():
	"""
	Early stopping to stop the training when the loss does not improve after
	certain epochs.
	"""
	def __init__(self, patience=3, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is
			   not improving
		:param min_delta: minimum difference between new loss and old loss for
			   new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, test_loss):
		if self.best_loss == None:
			self.best_loss = test_loss
		elif self.best_loss - test_loss > self.min_delta:
			self.best_loss = test_loss
		elif self.best_loss - test_loss < self.min_delta:
			self.counter += 1
			# print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				print('INFO: Early stopping')
				self.early_stop = True


class EarlyStoppingBatch():
	"""
	Like EarlyStopping, but stops when all members of batch meet the individual stopping criteria.
	Note: this is used for attacks, so loss is being maximized
	"""
	def __init__(self, bs, patience=3, min_delta=1e-1):
		"""
		:param patience: how many epochs to wait before stopping when loss is
			   not improving
		:param min_delta: minimum difference between new loss and old loss for
			   new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta

		self.counter = torch.zeros(bs)
		self.best_loss = None #torch.ones(bs)*float("inf")
		self.early_stop_vec = torch.zeros(bs, dtype=torch.bool)
		self.early_stop = False

	def __call__(self, test_loss):
		if self.best_loss == None:
			self.best_loss = test_loss

		# print("Inside EarlyStoppingBatch")
		# IPython.embed()

		# print(self.best_loss, test_loss)
		improve_ind = torch.nonzero(test_loss - self.best_loss >= self.min_delta)
		nonimprove_ind = torch.nonzero(test_loss - self.best_loss < self.min_delta)
		self.best_loss[improve_ind] = test_loss[improve_ind]

		self.counter[nonimprove_ind] = self.counter[nonimprove_ind] + 1

		early_stop_ind = torch.nonzero(self.counter >= self.patience)
		self.early_stop_vec[early_stop_ind] = True

		# print(self.counter)
		if torch.all(self.early_stop_vec).item():
			print('INFO: Early stopping')
			self.early_stop = True

class IndexNNInput(nn.Module):
	def __init__(self, which_ind):
		"""
		:param which_ind: flat numpy array
		"""
		self.which_ind = which_ind
		self.output_dim = len(which_ind)

	def forward(self, x):
		return x[:, self.which_ind]


class TransformEucNNInput(nn.Module):
	# Note: this is specific to FlyingInvPend
	def __init__(self, state_index_dict):
		"""
		:param which_ind: flat numpy array
		"""
		super().__init__()
		self.state_index_dict = state_index_dict
		self.output_dim = 12

	def forward(self, x):
		alpha = x[:, self.state_index_dict["alpha"]]
		beta = x[:, self.state_index_dict["beta"]]
		gamma = x[:, self.state_index_dict["gamma"]]

		dalpha = x[:, self.state_index_dict["dalpha"]]
		dbeta = x[:, self.state_index_dict["dbeta"]]
		dgamma = x[:, self.state_index_dict["dgamma"]]

		phi = x[:, self.state_index_dict["phi"]]
		theta = x[:, self.state_index_dict["theta"]]

		dphi = x[:, self.state_index_dict["dphi"]]
		dtheta = x[:, self.state_index_dict["dtheta"]]

		# print("inside TransformEucNNInput's forward()")
		# IPython.embed()

		x_quad = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)
		y_quad = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		z_quad = torch.cos(beta)*torch.cos(gamma)

		d_x_quad_d_alpha = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		d_x_quad_d_beta = -torch.cos(alpha)*torch.cos(beta)*torch.cos(gamma)
		d_x_quad_d_gamma = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		v_x_quad = dalpha*d_x_quad_d_alpha + dbeta*d_x_quad_d_beta + dgamma*d_x_quad_d_gamma

		d_y_quad_d_alpha = -torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.sin(alpha)*torch.sin(gamma)
		d_y_quad_d_beta = -torch.sin(alpha)*torch.cos(beta)*torch.cos(gamma)
		d_y_quad_d_gamma = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		v_y_quad = dalpha*d_y_quad_d_alpha + dbeta*d_y_quad_d_beta + dgamma*d_y_quad_d_gamma

		v_z_quad = dbeta*torch.sin(beta)*torch.cos(gamma) + dgamma*torch.cos(beta)*torch.sin(gamma)

		x_pend = torch.sin(theta)*torch.cos(phi)
		y_pend = -torch.sin(phi)
		z_pend = torch.cos(theta)*torch.cos(phi)

		v_x_pend = -dtheta*torch.cos(theta)*torch.cos(phi) + dphi*torch.sin(theta)*torch.sin(phi)
		v_y_pend = dphi*torch.cos(phi)
		v_z_pend = dtheta*torch.sin(theta)*torch.cos(phi) + dphi*torch.cos(theta)*torch.sin(phi)

		rv = torch.cat([x_quad[:, None], y_quad[:, None], z_quad[:, None], v_x_quad[:, None], v_y_quad[:, None], v_z_quad[:, None], x_pend[:, None], y_pend[:, None], z_pend[:, None], v_x_pend[:, None], v_y_pend[:, None], v_z_pend[:, None]], dim=1)
		return rv
