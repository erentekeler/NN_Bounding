'''This script runs the bound propagation and gurobi models for the example given in the appendix of https://arxiv.org/pdf/2002.12920'''

import torch 
from torch import nn

from src.gurobi_verifiers.gurobi_LP import solve_LP
from src.bound_prop.IBP import IBP
from src.bound_prop.backward_lirpa import backward_lirpa
from src.bound_prop.forward_lirpa import forward_lirpa



'''Auto lirpa paper model'''
NN = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)


'''Loading the given model weights and biases'''
# I took the parameters from the paper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN
model[0].weight = nn.Parameter(torch.tensor([[2, 1], [-3, 4]], dtype=torch.float32))
model[0].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

model[2].weight = nn.Parameter(torch.tensor([[4, -2], [2, 1]], dtype=torch.float32))
model[2].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

model[4].weight = nn.Parameter(torch.tensor([[-2, 1]], dtype=torch.float32))
model[4].bias = nn.Parameter(torch.tensor([0], dtype=torch.float32))

model.to(device)



'''Paper has given relaxation parameters, you can load them as follows'''
# dictionary indices denote the layer index, 1st and 3rd layers are ReLUs for the given NN
# lb relaxations are already 0x+0 by default, paper assumed the same
ub_relaxations = {1:{"Upper_bound_slope": torch.tensor([0.58, 0.64]),
                     "Upper_bound_bias": torch.tensor([2.92, 6.43])},
                  3:{"Upper_bound_slope": torch.tensor([0.4375, 1]),
                     "Upper_bound_bias": torch.tensor([15.75, 0]),
                     "Lower_bound_slope": torch.tensor([0, 1.0])}}



# Setting the verification domain 
input_range = torch.tensor([[-2, 2], [-1, 3]]).to(device)

# IBP
IBP = IBP(model, input_range=input_range, c=None)
IBP.compute_bounds(print_interm_bounds=False, print_out_bounds=True)

# Backward
backward_lirpa = backward_lirpa(model=model, input_range=input_range, ub_relaxations=ub_relaxations, c=None)
backward_lb, backward_ub = backward_lirpa.compute_bounds(print_out_bounds=True)

# Forward
forward_lirpa = forward_lirpa(model=model, input_range=input_range, ub_relaxations=ub_relaxations, c=None)
forward_lb, forward_ub = forward_lirpa.compute_bounds(print_out_bounds=True)

# Triangular relaxation LP
solve_LP(model, input_range=input_range, model_type="triangular", c=None)

# MILP
solve_LP(model, input_range=input_range, model_type="MILP", c=None)