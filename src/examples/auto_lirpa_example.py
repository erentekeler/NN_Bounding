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




# Setting the verification domain 
input_range = torch.tensor([[-2, 2], [-1, 3]]).to(device)

'''I demonstrate 2 setups here, the first one is loading the relaxation parameters given in the paper'''
# dictionary indices denote the layer index, 1st and 3rd layers are ReLUs for the given NN
ub_relaxations = {1:{"custom_ub_slope": torch.tensor([0.58, 0.64], dtype=torch.float32),
                     "custom_ub_bias": torch.tensor([2.92, 6.43], dtype=torch.float32)},
                  3:{"custom_ub_slope": torch.tensor([0.4375, 1], dtype=torch.float32),
                     "custom_ub_bias": torch.tensor([15.75, 0], dtype=torch.float32)}}

lb_relaxations = {1:{"custom_lb_slope": torch.tensor([0, 0], dtype=torch.float32),
                     "custom_lb_bias": torch.tensor([0, 0], dtype=torch.float32)},
                  3:{"custom_lb_slope": torch.tensor([0, 1.0], dtype=torch.float32),
                     "custom_lb_bias": torch.tensor([0, 0], dtype=torch.float32)}}


# IBP
ibp = IBP(model, input_range=input_range, c=None, compute_relaxation_params=False) # No need to compute slopes and biases
ibp_lb, ibp_ub = ibp.compute_bounds(print_interm_bounds=False, print_out_bounds=True)

# Backward
backward = backward_lirpa(model=model, input_range=input_range, ub_relaxations=ub_relaxations, lb_relaxations=lb_relaxations, c=None, relaxation_method="custom", compute_interm_bounds=True)
backward_lb, backward_ub = backward.compute_bounds(print_out_bounds=True)

# Forward
forward = forward_lirpa(model=model, input_range=input_range, ub_relaxations=ub_relaxations, lb_relaxations=lb_relaxations, c=None, relaxation_method="custom")
forward_lb, forward_ub = forward.compute_bounds(print_out_bounds=True)

# Triangular relaxation LP
solve_LP(model, input_range=input_range, model_type="triangular", c=None, interm_method="backward", relaxation_method="custom", layer_information=backward.layer_information)

# MILP
solve_LP(model, input_range=input_range, model_type="MILP", c=None, interm_method="backward", relaxation_method="custom", layer_information=backward.layer_information)




'''The relaxations that are used in backward are computed via forward, we can directly mimic that '''
# IBP
ibp = IBP(model, input_range=input_range, c=None, compute_relaxation_params=False) # No need to compute slopes and biases
ibp_lb, ibp_ub = ibp.compute_bounds(print_interm_bounds=False, print_out_bounds=True)

# Forward
forward = forward_lirpa(model=model, input_range=input_range, ub_relaxations=ub_relaxations, lb_relaxations=lb_relaxations, c=None, relaxation_method="forward")
forward_lb, forward_ub = forward.compute_bounds(print_out_bounds=True)
ub_relaxation, lb_relaxations = forward.export_relaxation_params()

# Backward
backward = backward_lirpa(model=model, input_range=input_range, ub_relaxations=ub_relaxations, lb_relaxations=lb_relaxations, c=None, relaxation_method="custom", compute_interm_bounds=True)
backward_lb, backward_ub = backward.compute_bounds(print_out_bounds=True)
