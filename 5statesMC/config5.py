import numpy as np
import torch
eps = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
stationary_pi = torch.tensor([[0.19739931], [0.22828246], [0.22340618], [0.2080549] , [0.14285714]]).to(device)

coeff_midpoint1 = torch.tensor([[0.35,0.35,0.1,0.1,0.1]]).to(device)
invert_coeff1 = torch.log(coeff_midpoint1/torch.max(coeff_midpoint1))

coeff_midpoint2 = torch.tensor([[0.1,0.35,0.35,0.1,0.1]]).to(device)
invert_coeff2 = torch.log(coeff_midpoint2/torch.max(coeff_midpoint2))

coeff_midpoint3 = torch.tensor([[0.1,0.1,0.35,0.35,0.1]]).to(device)
invert_coeff3 = torch.log(coeff_midpoint3/torch.max(coeff_midpoint3))

v1 = torch.tensor([[1./0.19739931,0.0,0.0,0.0,0.0]]).to(device)
v2 = torch.tensor([[0.0,1./0.22828246,0.0,0.0,0.0]]).to(device)
v3 = torch.tensor([[0.0,0.0,1./0.22340618,0.0,0.0]]).to(device)
v4 = torch.tensor([[0.0,0.0,0.0,1./0.2080549,0.0]]).to(device)
v5 = torch.tensor([[0.0,0.0,0.0,0.0,1./0.14285714]]).to(device)
v = torch.cat([v1,v2,v3,v4,v5],0)
'''
Transition matrix in Markov Chain Q
Q(x,y) = Prob from going from state x to state y
'''
transition_matrix =  torch.tensor([[0.3, 0.3, 0.1, 0.2, 0.1],
                                   [0.2, 0.3, 0.2, 0.2, 0.1],
                                   [0.1, 0.2, 0.3, 0.3, 0.1],
                                   [0.2, 0.2, 0.3, 0.2, 0.1],
                                   [0.2, 0.1, 0.2, 0.1, 0.4]]).to(device)
ndim = 5