import torch
import numpy as np

from gurobi_LP import solve_LP
from NN_model import NeuralNetwork
from IBP import IBP

'''Defining the model and the input range'''
# %% This part initializes the model and solves for the convex relaxations
# Fix the seed and initialize the model
torch.manual_seed(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().NN.to(device)

# I determine the input shape based on model parameters to be generic
# i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validity of the bounds
input_size = model[0].weight.shape[1]
input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

# Here I am creating the linear coefficients for the property we want to verify
# I implemented it in a way that it computes elementwise bounds when c = "all"
# output_size = model[-1].weight.shape[0]
# c = np.zeros(output_size)
# c[15] = 1
# c[16] = 1

IBP = IBP(model, input).print_IBP_results(verbose=False)
solve_LP(model, input, model_type="triangular", c="all")
solve_LP(model, input, model_type="MILP", c="all")
# solve_LP(model, input, model_type="MILP")