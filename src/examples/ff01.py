'''This script runs the bound propagation and gurobi models for a sequential NN.'''

import torch 
from torch import nn

from src.gurobi_verifiers.gurobi_LP import solve_LP
from src.bound_prop.IBP import IBP
from src.bound_prop.backward_lirpa import backward_lirpa
from src.bound_prop.forward_lirpa import forward_lirpa

# Fixing the random seed for reproducability
torch.manual_seed(10)

'''Creating the model'''
NN = nn.Sequential(
    nn.Linear(20, 15),
    nn.ReLU(),
    nn.Linear(15, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
    nn.ReLU(),
    nn.Linear(3, 3)
)


'''Loading the model and passing it to GPU'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN.to(device)

'''Verification settings'''
# Setting the verification domain 
input_size = model[0].weight.shape[1]
input_range = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

# Creating a linear property vector out of ones
# If you want to compute elementwise bounds, set c=None
output_size = model[-1].weight.shape[0]
c = torch.ones((output_size, 1), device=device) # Must be in the matrix form (size, 1)
# c[2] = -1 # you can set the elements of the property vector here
# c=None

'''Running the bound prop algorithms and Gurobi'''
# Backward
backward = backward_lirpa(model=model, input_range=input_range, c=None, relaxation_method="backward", compute_interm_bounds=True)
backward_lb, backward_ub = backward.compute_bounds(print_out_bounds=True)

# Forward
forward = forward_lirpa(model=model, input_range=input_range, c=None, relaxation_method="IBP")
forward_lb, forward_ub = forward.compute_bounds(print_out_bounds=True)

# IBP
ibp = IBP(model, input_range=input_range, c=None, compute_relaxation_params=True) # No need to compute slopes and biases
ibp_lb, ibp_ub = ibp.compute_bounds(print_interm_bounds=False, print_out_bounds=True)

# Triangular relaxation LP
solve_LP(model, input_range=input_range, model_type="triangular", c=None, interm_method="IBP", relaxation_method="IBP", layer_information=ibp.layer_information)

# MILP
solve_LP(model, input_range=input_range, model_type="MILP", c=None, interm_method="IBP", relaxation_method="IBP", layer_information=ibp.layer_information)