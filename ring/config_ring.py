import numpy as np
import torch
eps = 1e-10
ndim = 5
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
P = np.array([
    [0.0, 1./2, 0., 0.,1./2],
    [1./2, 0.0, 1./2, 0.0, 0.0],
    [0.0, 1./2, 0.0, 1./2,0.0],
    [0.0, 0.0, 1./2, 0.0, 1./2],
    [1./2, 0.0, 0.0, 1./2, 0.0]
])

# Compute stationary distribution
eigvals, eigvecs = np.linalg.eig(P.T)
stat_index = np.argmin(np.abs(eigvals - 1))
stat_dist = np.real(eigvecs[:, stat_index])
stat_dist /= np.sum(stat_dist)
stationary_pi = torch.from_numpy(stat_dist).reshape(ndim,1).float().to(device)

coeff_midpoint1 = torch.tensor([[0.4,0.15,0.15,0.15,0.15]]).to(device)
invert_coeff1 = torch.log(coeff_midpoint1/torch.max(coeff_midpoint1))

coeff_midpoint2 = torch.tensor([[0.15,0.15,0.4,0.15,0.15]]).to(device)
invert_coeff2 = torch.log(coeff_midpoint2/torch.max(coeff_midpoint2))

coeff_midpoint3 = torch.tensor([[0.2,0.2,0.2,0.2,0.2]]).to(device)
invert_coeff3 = torch.log(coeff_midpoint3/torch.max(coeff_midpoint3))

v1 = torch.tensor([[1./stat_dist[0],0.0,0.0,0.0,0.0]]).float().to(device)
v2 = torch.tensor([[0.0,1./stat_dist[1],0.0,0.0,0.0]]).float().to(device)
v3 = torch.tensor([[0.0,0.0,1./stat_dist[2],0.0,0.0]]).float().to(device)
v4 = torch.tensor([[0.0,0.0,0.0,1./stat_dist[3],0.0]]).float().to(device)
v5 = torch.tensor([[0.0,0.0,0.0,0.0,1./stat_dist[4]]]).float().to(device)
v = torch.cat([v1,v2,v3,v4,v5],0)
'''
Transition matrix in Markov Chain Q
Q(x,y) = Prob from going from state x to state y
'''
transition_matrix =  torch.from_numpy(P).float().reshape(1,ndim,ndim).to(device)

adjacency_matrix = torch.ceil(transition_matrix)
pi_q = torch.tile(stationary_pi.reshape(ndim,1),(1,ndim))*transition_matrix