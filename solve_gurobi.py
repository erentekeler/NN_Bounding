import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np
import torch
from torch import nn

from Bounding import Bounding
from NN_model import NeuralNetwork

# %% This part initializes the model and solves for the convex relaxations
# Fix the seed and initialize the model
torch.manual_seed(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)

# I determine the input shape based on model parameters to be generic
# i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
input_size = model.NN[0].weight.shape[1]
input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

Bounding = Bounding(model, input)
layer_information = Bounding.layer_information 

# %% Implemented the Gurobi model here
# Creating the model
m = gp.Model()

# The input, assume bounded by inf norm, bounds are explicitly given
N = input.shape[0] # Number of rows in the input
input_var = m.addMVar(N, vtype=GRB.CONTINUOUS, name="input", lb=input[:, 0].cpu().numpy(), ub=input[:, 1].cpu().numpy()) 

expression = input_var # The inital expression is the input_var
for layer_idx, layer in layer_information.iterrows():
    # If the layer is linear, just compute an expression consists of weight, bias and previous expression
    if layer["Layer_type"] == "nn.Linear":
        # Getting the linear layer weights and biases to compute the layer output
        expression = model.NN[layer_idx].weight.detach().cpu().numpy() @ expression + model.NN[layer_idx].bias.detach().cpu().numpy() 
   
   # If the layer is a ReLU, constrain it between relaxations
    elif layer["Layer_type"] == "nn.ReLU":
        n_neurons = layer["Upper_bound_slope"].shape[0] # Get the number of neurons

        # Creating the post activation constraint for relaxed ReLU functions
        post_activation = m.addMVar(n_neurons, vtype=GRB.CONTINUOUS, name=f'z_hat_{layer_idx}')

        # Getting the relaxation parameters
        upper_slope = layer["Upper_bound_slope"].detach().cpu().numpy()
        upper_bias = layer["Upper_bound_bias"].detach().cpu().numpy()

        lower_slope = layer["Lower_bound_slope"].detach().cpu().numpy()
        lower_bias = layer["Lower_bound_bias"].detach().cpu().numpy()

        # Post activation constraints, 
        m.addConstr(post_activation >= lower_slope * expression + lower_bias)
        m.addConstr(post_activation <= upper_slope * expression + upper_bias)

        expression = post_activation # The expression is flowing through the layers



# Objective function is the last expression
m.setObjective(expression, gp.GRB.MINIMIZE)

m.optimize()

print(f"Optimal objective value: {m.objVal}")
print(f"Solution values: x= \n {input_var.X.reshape(-1, 1)}")


