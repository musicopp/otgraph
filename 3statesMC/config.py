import numpy as np
import torch
eps = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
stationary_pi = torch.tensor([[2./9.],[1./3.],[4./9.]]).to(device)

# Initialize boundary conditions
# Boundary conditions
x_1 = torch.tensor([[0.45*4.5, 0.45*3.,0.1*9./4.]]).to(device)
x_2 = torch.tensor([[0.1*4.5, 0.45*3.,0.45*9./4.]]).to(device)
x_3 = torch.tensor([[0.45*4.5, 0.1*3.,0.45*9./4.]]).to(device)

coeff_midpoint1 = torch.tensor([[0.45,0.45,0.1]]).to(device)
invert_coeff1 = torch.log(coeff_midpoint1/torch.max(coeff_midpoint1))

coeff_midpoint2 = torch.tensor([[0.1,0.45,0.45]]).to(device)
invert_coeff2 = torch.log(coeff_midpoint2/torch.max(coeff_midpoint2))

coeff_midpoint3 = torch.tensor([[0.45,0.1,0.45]]).to(device)
invert_coeff3 = torch.log(coeff_midpoint3/torch.max(coeff_midpoint3))
'''
Transition matrix in Markov Chain Q
Q(x,y) = Prob from going from state x to state y
'''
transition_matrix = torch.tensor([[0.0  ,1./4.,3./4.],
								  [1./6.,0.0  ,5./6.],
								  [3./8.,5./8.,0.0]]).to(device)
pi_q = torch.tensor([[0.0*2/9  ,1./4.*2/9,3./4.*2/9],
					 [1./(6.*3),0.0  ,5./(6.*3)],
					 [12./72.,20./72.,0.0]]).to(device)

ndim = 3

theta_gradpsi12 = torch.tensor([[0.0,0.0,-21.0/10],
							  [0.0,0.0  ,0.0],
							  [21.0/10,0.0,0.0]]).to(device)
theta_gradpsi23 = torch.tensor([[0.0,6.3,0.0],
							  [-6.3,0.0  ,0.0],
							  [0.0,0.0,0.0]]).to(device)
theta_gradpsi31 = torch.tensor([[0.0,0.0,0.0],
							  [0.0,0.0,1.26],
							  [0.0,-1.26,0.0]]).to(device)