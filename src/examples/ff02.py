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
    nn.Linear(500, 300),
    nn.ReLU(),
    nn.Linear(300, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 20)
)


'''Loading the model and passing it to GPU'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN.to(device)

'''Verification settings-This example is with a pure norm ball'''
# Setting the verification domain     
input_size = model[0].weight.shape[1]
x_0 = torch.ones(input_size, dtype=torch.float32, device=device)
norm = 2
eps = 2

# Creating a linear property vector out of ones
# If you want to compute elementwise bounds, set c=None
output_size = model[-1].weight.shape[0]
c = torch.zeros((output_size, 1), device=device) # Must be in the matrix form (size, 1)
c[2] = -1 # you can set the elements of the property vector here
c[3] = 2 # you can set the elements of the property vector here
# c=None

'''Running the bound prop algorithms and Gurobi'''
# IBP
IBP = IBP(model, x_0=x_0, norm=norm, eps=eps, c=c)
IBP.compute_bounds(print_interm_bounds=False, print_out_bounds=True)

# Backward
backward_lirpa = backward_lirpa(model=model, x_0=x_0, norm=norm, eps=eps, c=c, relaxation_method="backward", compute_interm_bounds=True)
backward_lb, backward_ub = backward_lirpa.compute_bounds(print_out_bounds=True)

# Forward
forward_lirpa = forward_lirpa(model=model, x_0=x_0, norm=norm, eps=eps, c=c, relaxation_method="forward", compute_interm_bounds=True)
forward_lb, forward_ub = forward_lirpa.compute_bounds(print_out_bounds=True)


'''These are not implemented for input being a pure norm ball'''
# Triangular relaxation LP
# c = c.detach().cpu().numpy().flatten()
# solve_LP(model, input_range=input_range, model_type="triangular", c=c)

# MILP, fairly larger model, hard to solve with binaries
# solve_LP(model, input_range=input_range, model_type="MILP", c=None)