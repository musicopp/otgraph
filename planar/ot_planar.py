#import torch
import torch.nn as nn
from config_planar import *

def projection_p(p):
	'''
	Input:  vector p: bz x ndim
	Output: vector rho: bz x ndim 
			s.t. < rho , stationary_pi > = 1 
	'''
	p_pos = p**2
	dot_product = p_pos@stationary_pi # bz x 1
	return p_pos*dot_product/(dot_product**2+eps)

def div_grad(mean,V):
	'''
	Input:  mean: bz x ndim x dim
			grad vector V at time t: bz x ndim x ndim
	Output: Div (grad psi) defined by
			Div (grad psi x) = sum_{y} grad psi(x,y)
			Div : bz x ndim
	'''
	return torch.sum(mean*V*transition_matrix,2)

def grad_psi(psi):
	'''
	Input: psi: bz x ndim

	Output: matrix A: bz x ndim x ndim
					  A(t)_{ij} is psi_t(x_j)-psi_t(x_i)
	'''
	psi_reshape=torch.unsqueeze(psi,1)
	psi_trans = torch.transpose(psi_reshape,2,1)
	a_col = torch.tile(psi_reshape,(1,ndim,1))
	a_row = torch.transpose(a_col,1,2)
	return (a_col - a_row)*adjacency_matrix

def logarithmic_mean(rho):

	rho_reshape = torch.unsqueeze(rho,1)
	x = torch.tile(rho_reshape,(1,ndim,1)) # column
	y = torch.transpose(x,1,2) # row
	#x = torch.as_tensor(x, dtype=torch.float64)
	#y = torch.as_tensor(y, dtype=torch.float64)

	if torch.any(x < 0) or torch.any(y < 0):
		raise ValueError("Inputs must be non-negative")
	same_mask = ((x - y)**2 < eps)
	#xzero_mask = (x**2 < eps)
	#yzero_mask = (y**2 < eps)
	mask = same_mask #| xzero_mask | yzero_mask
	diff_mask = ~mask

	log_mean = torch.empty_like(x)
	dlog_d1 = torch.empty_like(x)

	#log_mean[xzero_mask] = 0.0
	#log_mean[yzero_mask] = 0.0
	log_mean[same_mask] = x[same_mask]

	dlog_d1[same_mask] = 0.5
	#dlog_d1[xzero_mask] = 0.0  # infinity
	#dlog_d1[yzero_mask] = 0.0
	

	x_mask = x[diff_mask]
	y_mask = y[diff_mask]
	log_x = torch.log(x_mask)
	log_y = torch.log(y_mask)

	log_mean[diff_mask] = (x_mask - y_mask) / (log_x-log_y)
	dlog_d1[diff_mask] =  ((log_x-log_y)*x_mask-(x_mask-y_mask))/(((log_x-log_y)**2)*x_mask+eps)
	
	return log_mean,dlog_d1

def pde(t,net_rho,net_psi,loss):
	rho = net_rho(t)
	psi = net_psi(t)
	bz = t.shape[0]
	all_zero = torch.zeros((bz,ndim)).to(device)
	drho_dt = torch.zeros((bz,ndim)).to(device)
	dpsi_dt = torch.zeros((bz,ndim)).to(device)
	for j in range(ndim):
		tangent = torch.zeros_like(rho)
		tangent[:,j]=1
		drho_dt[:,j] = torch.autograd.grad(rho, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	for j in range(ndim):
		tangent = torch.zeros_like(psi)
		tangent[:,j]=1
		dpsi_dt[:,j] = torch.autograd.grad(psi, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	logmean,dmean_d1 = logarithmic_mean(rho)
	
	gradpsi = grad_psi(psi)
	divgrad = div_grad(logmean,gradpsi)

	# Continuitiy equation
	# http://etheses.dur.ac.uk/14124/1/Kamtue_thesis.pdf?DDD21+ 
	# page 158-159 Eq. 21.1
	pde1 = loss(drho_dt+divgrad,all_zero)

	# Geodesics equation
	# http://etheses.dur.ac.uk/14124/1/Kamtue_thesis.pdf?DDD21+ 
	# page 161 Eq. 21.3
	pde2 = loss(dpsi_dt + torch.sum( (gradpsi**2)*transition_matrix*dmean_d1,2)/2.0,all_zero)
	
	return pde1,pde2
def entropy(rho):
	'''
	Input:  rho: bz x ndim
	Output: entropy: bz x 1
	'''
	return torch.sum(rho*torch.log(rho)*(stationary_pi.T),1,keepdim=True)
def softmax_list(x):
    """Compute softmax from a list input and return a list output."""
    x = np.array(x)
    e_x = np.exp(x - np.max(x))  # for numerical stability
    softmax_values = e_x / e_x.sum()
    return softmax_values.tolist()
def compute_speed(rho,psi):
	'''
	Input:  rho: bz x ndim
			psi: bz x ndim
	Output: flow: bz x ndim
	'''
	logmean,dmean_d1 = logarithmic_mean(rho)
	gradpsi = grad_psi(psi)
	stationary_pi_reshape = torch.tile(stationary_pi,(1,3)).reshape(1,3,3) # bz x 3 x 3
	speed = torch.sqrt(torch.sum((gradpsi**2)*logmean*transition_matrix*stationary_pi_reshape,dim=(1, 2))/2) # bz x 3 x 3
	return speed
def compute_flow(rho,psi):
	'''
	Input:  rho: bz x ndim
			psi: bz x ndim
	Output: flow: bz x ndim
	'''
	logmean,dmean_d1 = logarithmic_mean(rho)
	gradpsi = grad_psi(psi)
	flow = pi_q*gradpsi*logmean
	return flow

def compute_distance(rho,psi):
	'''
	Input:  rho: bz x ndim
			psi: bz x ndim
	Output: flow: bz x ndim
	'''
	logmean,dmean_d1 = logarithmic_mean(rho)
	gradpsi = grad_psi(psi)
	stationary_pi_reshape = torch.tile(stationary_pi,(1,3)).reshape(1,3,3) # bz x 3 x 3
	distance = torch.mean(torch.sum((gradpsi**2)*logmean*transition_matrix*stationary_pi_reshape,dim=(1, 2))/2) # bz x 3 x 3
	return distance