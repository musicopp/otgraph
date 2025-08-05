import numpy as np
import torch
eps = 1e-10
ndim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
stationary_pi = torch.tensor([[1./2.],[1./2.]]).to(device)

coeff_midpoint1 = torch.tensor([[0.6,0.4]]).to(device)
invert_coeff1 = torch.log(coeff_midpoint1/torch.max(coeff_midpoint1))

coeff_midpoint2 = torch.tensor([[0.5,0.5]]).to(device)
invert_coeff2 = torch.log(coeff_midpoint2/torch.max(coeff_midpoint2))

coeff_midpoint3 = torch.tensor([[0.7,0.3]]).to(device)
invert_coeff3 = torch.log(coeff_midpoint3/torch.max(coeff_midpoint3))
v1 = torch.tensor([[1./2,0.0]]).float().to(device)
v2 = torch.tensor([[0.0,1./2]]).float().to(device)
v = torch.cat([v1,v2],0)
'''
Transition matrix in Markov Chain Q
Q(x,y) = Prob from going from state x to state y
'''
transition_matrix = torch.tensor([[0.0 ,1.],
                                  [1., 0.0 ]]).to(device)

pi_q = torch.tile(stationary_pi.reshape(ndim,1),(1,ndim))*transition_matrix