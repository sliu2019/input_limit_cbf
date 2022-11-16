import torch
import numpy as np

from torch import nn
import os, sys
import IPython
import math

g = 9.81
class RhoMax(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		# print("Inside RhoMax forward")
		# IPython.embed()
		theta = x[:, [self.i["theta"]]]
		phi = x[:, [self.i["phi"]]]
		gamma = x[:, [self.i["gamma"]]]
		beta = x[:, [self.i["beta"]]]

		cos_cos = torch.cos(theta)*torch.cos(phi)
		eps = 1e-4 # prevents nan when cos_cos = +/- 1 (at x = 0)
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps
		delta = torch.acos(cos_cos + signed_eps)
		rv = torch.maximum(torch.maximum(delta**2, gamma**2), beta**2) - self.delta_safety_limit**2
		return rv

class RhoSum(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		# print("Inside RhoSum forward")
		# IPython.embed()
		theta = x[:, [self.i["theta"]]]
		phi = x[:, [self.i["phi"]]]
		gamma = x[:, [self.i["gamma"]]]
		beta = x[:, [self.i["beta"]]]

		cos_cos = torch.cos(theta)*torch.cos(phi)
		eps = 1e-4 # prevents nan when cos_cos = +/- 1 (at x = 0)
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps
		delta = torch.acos(cos_cos + signed_eps)
		rv = delta**2 + gamma**2 + beta**2 - self.delta_safety_limit**2

		return rv

class XDot(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device
		self.i = self.state_index_dict

	def forward(self, x, u):
		# x: bs x 10, u: bs x 4
		# The way these are implemented should be batch compliant

		# Pre-computations
		# Compute the rotation matrix from quad to global frame
		# Extract the k_{x,y,z}
		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		phi = x[:, self.i["phi"]]
		theta = x[:, self.i["theta"]]
		dphi = x[:, self.i["dphi"]]
		dtheta = x[:, self.i["dtheta"]]

		R = torch.zeros((x.shape[0], 3, 3), device=self.device) # is this the correct rotation?
		R[:, 0, 0] = torch.cos(alpha)*torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)
		R[:, 1, 0] = torch.sin(alpha)*torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta)*torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta)*torch.cos(gamma)

		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		F = (u[:, 0] + self.M*g)

		###### Computing state derivatives
		# IPython.embed()
		J = torch.tensor([self.J_x, self.J_y, self.J_z]).to(self.device)
		norm_torques = u[:, 1:]*(1.0/J)

		ddquad_angles = torch.bmm(R, norm_torques[:, :, None]) # (N, 3, 1)
		ddquad_angles = ddquad_angles[:, :, 0]
		# ddgamma = (1.0/self.J_x)*ddquad_angles[:, 0]
		# ddbeta = (1.0/self.J_y)*ddquad_angles[:, 1]
		# ddalpha = (1.0/self.J_z)*ddquad_angles[:, 2]

		ddgamma = ddquad_angles[:, 0]
		ddbeta = ddquad_angles[:, 1]
		ddalpha = ddquad_angles[:, 2]

		ddphi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dtheta*dphi*torch.tan(theta)
		ddtheta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dphi)*torch.sin(theta)*torch.cos(theta)

		# Excluding translational motion
		rv = torch.cat([x[:, [self.i["dgamma"]]], x[:, [self.i["dbeta"]]], x[:, [self.i["dalpha"]]], ddgamma[:, None], ddbeta[:, None], ddalpha[:, None], dphi[:, None], dtheta[:, None], ddphi[:, None], ddtheta[:, None]], axis=1)
		return rv

class ULimitSetVertices(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device

		# precompute, just tile it in forward()
		k1 = self.k1
		k2 = self.k2
		l = self.l
		M = np.array([[k1, k1, k1, k1], [0, -l*k1, 0, l*k1], [l*k1, 0, -l*k1, 0], [-k2, k2, -k2, k2]]) # mixer matrix

		r1 = np.concatenate((np.zeros(8), np.ones(8)))
		r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
		r3 = np.concatenate((np.zeros(2), np.ones(2),np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),np.zeros(2), np.ones(2)))
		r4 = np.zeros(16)
		r4[1::2] = 1.0
		impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0) # 16 vertices in the impulse control space

		# print("ulimitsetvertices")
		# IPython.embed()

		force_vert = M@impulse_vert - np.array([[self.M*g], [0.0], [0.0], [0.0]]) # Fixed bug: was subtracting self.M*g (not just in the first row)
		force_vert = force_vert.T.astype("float32")
		self.vert = torch.from_numpy(force_vert).to(self.device)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# (bs, n_vertices, u_dim) or (bs, 16, 4)

		rv = self.vert
		rv = rv.unsqueeze(dim=0)
		rv = rv.expand(x.shape[0], -1, -1)
		return rv