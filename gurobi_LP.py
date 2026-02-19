'''This model is the LP relaxation of a ReLU NN. 
1) It employes IBP first, then sets triangular relaxations for ReLU activations accordingly.'''

import gurobipy as gp

import pandas as pd
import numpy as np
import torch
from torch import nn

from Bounding import Bounding
from NN_model import NeuralNetwork
from gurobi_helper import constrain_ReLU


def solve_LP(model, input_range, model_type, c=None):
    # Computing the IBP bounds, and getting the network information
    bound_computer = Bounding(model, input_range)
    layer_information = bound_computer.layer_information 

    # %% Implemented the Gurobi model here
    # Creating the model
    m = gp.Model()
    m.params.OutputFlag = 0

    # The input, assume bounded by inf norm, bounds are explicitly given
    N = input_range.shape[0] # Number of rows in the input
    input = m.addMVar(N, lb=input_range[:, 0].cpu().numpy(), ub=input_range[:, 1].cpu().numpy(), name="x0") 

    layer_input = input # The inital layer_input is the input
    for layer_idx, layer in layer_information.iterrows():
        # If the layer is linear, compute the layer output just using weight, bias and layer_input
        if layer["Layer_type"] == "nn.Linear":
            out_dim = model[layer_idx].weight.shape[0] # Getting the number of output neurons

            # Defining the layer output variable
            layer_output = m.addMVar(out_dim, lb=layer["Layer_output"][:, 0], ub=layer["Layer_output"][:, 1], name=f'z_{layer_idx}') 

            # Getting the linear layer weights and biases to compute the layer output
            m.addConstr(layer_output == model[layer_idx].weight.detach().cpu().numpy() @ layer_input + model[layer_idx].bias.detach().cpu().numpy())

            if layer_idx == layer_information.index[-1]:
                final_layer_output = layer_output
            else:
                layer_input = layer_output # This then becomes a layer input to the next layer, an expression


    # If the layer is a ReLU, constrain it between relaxations
        elif layer["Layer_type"] == "nn.ReLU":
            layer_output = constrain_ReLU(m=m, layer=layer, layer_idx=layer_idx, layer_input=layer_input, model_type=model_type)  

            # This then becomes a layer input to the next layer
            if layer_idx == layer_information.index[-1]: # This is for the final layer
                final_layer_output = layer_output
            else:
                layer_input = layer_output # This then becomes a layer input to the next layer, an expression

    # get the lower bound
    m.setObjective(final_layer_output.sum(), gp.GRB.MINIMIZE)

    m.optimize()

    if m.Status == gp.GRB.OPTIMAL:
        lb = final_layer_output.X
        lb = [lb] if type(lb)==int else lb

    # get the upper bound
    m.setObjective(final_layer_output.sum(), gp.GRB.MAXIMIZE)

    m.optimize()

    if m.Status == gp.GRB.OPTIMAL:
        ub = final_layer_output.X
        ub = [ub] if type(ub)==int else ub

    # Print the output nicely
    print('************************************************************************', '\n')

    print('Gurobi Output Bounds: \n')
    
    for idx in range(len(lb)):
        print(f'{lb[idx]} <= f_{idx}(x) <= {ub[idx]}')

    print('\n************************************************************************', '\n')    

    # for v in m.getVars():
    #     print(f"{v.VarName}: {v.X}")


if __name__ == "__main__":

    # %% This part initializes the model and solves for the convex relaxations
    # Fix the seed and initialize the model
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().NN.to(device)

    # I determine the input shape based on model parameters to be generic
    # i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
    input_size = model[0].weight.shape[1]
    input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

    solve_LP(model, input, model_type="triangular")
