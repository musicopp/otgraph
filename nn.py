import torch
import torch.nn as nn
import numpy as np
from config import *
'''
Basic Neural Nets
'''
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
class basicNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,1024)
		self.linear_u_7 = nn.Linear(1024,1024)
		self.linear_u_8 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6,
				  self.linear_u_7,self.linear_u_8]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5,self.linear_u_6,self.linear_u_7]:
			u = self.activation_u(m(u))
		u = self.linear_u_8(u)
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u

class addNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.x_init = torch.tensor([[4.1500, 0.1000, 0.1000]]).to(device)
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,1024)
		self.linear_u_7 = nn.Linear(1024,1024)
		self.linear_u_8 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6,
				  self.linear_u_7,self.linear_u_8]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5,self.linear_u_6,self.linear_u_7]:
			u = self.activation_u(m(u))
		u = self.linear_u_8(u)

		u = self.x_init + inputs*u
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u
class barycentersine(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None,start_invert_coeff=invert_coeff1,end_invert_coeff=invert_coeff2):
		super().__init__()
		self.v1 = torch.tensor([[9./2.,0.0,0.0]]).to(device)
		self.v2 = torch.tensor([[0.0,3.0,0.0]]).to(device)
		self.v3 = torch.tensor([[0.0,0.0,9./4]]).to(device)
		self.v = torch.cat([self.v1,self.v2,self.v3],0)
		self.coeff_init = start_invert_coeff
		self.coeff_end = end_invert_coeff
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,1024)
		self.linear_u_7 = nn.Linear(1024,1024)
		self.linear_u_8 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6,
				  self.linear_u_7,self.linear_u_8]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5,self.linear_u_6,self.linear_u_7]:
			u = self.activation_u(m(u))
		u = self.linear_u_8(u)*torch.sin(np.pi*inputs) + self.coeff_init*torch.cos(np.pi*inputs/2)+self.coeff_end*torch.sin(np.pi*inputs/2)
		u = softmax(u)
		u = u@self.v
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u

class polyNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.x_init = torch.tensor([[4.1500, 0.1000, 0.1000]]).to(device)
		self.x_end = torch.tensor([[1.0,1.0,1.0]]).to(device)
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,1024)
		self.linear_u_7 = nn.Linear(1024,1024)
		self.linear_u_8 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6,
				  self.linear_u_7,self.linear_u_8]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5,self.linear_u_6,self.linear_u_7]:
			u = self.activation_u(m(u))
		u = self.linear_u_8(u)

		u = (1-inputs)**2*self.x_init + inputs**2*self.x_end + (1-inputs)*inputs*u
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u
class resNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,1024)
		self.linear_u_7 = nn.Linear(1024,1024)
		self.linear_u_8 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()
		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6]:
			torch.nn.init.xavier_uniform_(m.weight) 
	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5,self.linear_u_6,self.linear_u_7]:
			u = u+self.activation_u(m(u))
		u = self.linear_u_8(u)
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u
def softmax(x, dim=1):
    # Subtract the max value for numerical stability
    x_exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)+eps
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

class barycenterNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.v1 = torch.tensor([[9./2.,0.0,0.0]]).to(device)
		self.v2 = torch.tensor([[0.0,3.0,0.0]]).to(device)
		self.v3 = torch.tensor([[0.0,0.0,9./4]]).to(device)
		self.v = torch.cat([self.v1,self.v2,self.v3],0)
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,1024)
		self.linear_u_7 = nn.Linear(1024,1024)
		self.linear_u_8 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6,
				  self.linear_u_7,self.linear_u_8]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5,self.linear_u_6,self.linear_u_7]:
			u = self.activation_u(m(u))
		u = softmax(self.linear_u_8(u))
		u = u@self.v
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u

class barycenteraddNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None,invert_coeff=invert_coeff1):
		super().__init__()
		self.v1 = torch.tensor([[9./2.,0.0,0.0]]).to(device)
		self.v2 = torch.tensor([[0.0,3.0,0.0]]).to(device)
		self.v3 = torch.tensor([[0.0,0.0,9./4]]).to(device)
		self.v = torch.cat([self.v1,self.v2,self.v3],0)
		self.coeff_init = invert_coeff
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,1024)
		self.linear_u_7 = nn.Linear(1024,1024)
		self.linear_u_8 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6,
				  self.linear_u_7,self.linear_u_8]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5,self.linear_u_6,self.linear_u_7]:
			u = self.activation_u(m(u))
		u = self.linear_u_8(u)*inputs + self.coeff_init
		u = softmax(u)
		u = u@self.v
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u