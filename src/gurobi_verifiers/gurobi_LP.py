'''This model is the LP relaxation of a ReLU NN. 
1) It employes IBP first, then sets triangular relaxations for ReLU activations accordingly.'''

import gurobipy as gp

import pandas as pd
import numpy as np
import torch
from torch import nn

from src.Bounding import Bounding
from src.NN_model import NeuralNetwork
from src.gurobi_verifiers.gurobi_helper import constrain_ReLU


def solve_LP(model, input_range, model_type, c=None):
    # Computing the IBP bounds, and getting the network information
    bound_computer = Bounding(model, input_range=input_range)
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
            layer_output = m.addMVar(out_dim, lb=layer["Layer_output_bounds"][:, 0], ub=layer["Layer_output_bounds"][:, 1], name=f'z_{layer_idx}') 

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

    if c is None:
        output_dim = final_layer_output.shape[0]
        lb = np.zeros(output_dim)
        ub = np.zeros(output_dim)
        for i in range(output_dim):
            # Property function, as a linear combination
            # I am setting them to natural basis vectors to get the elementwise bounds 
            c = np.zeros(output_dim)
            c[i] = 1

            # get the lower bound
            m.setObjective(c.T@final_layer_output, gp.GRB.MINIMIZE)
            m.optimize()
            lb[i] = m.ObjVal

            # get the upper bound
            m.setObjective(c.T@final_layer_output, gp.GRB.MAXIMIZE)
            m.optimize()
            ub[i] = m.ObjVal
        
        # This is just to print elementwise bounds
        print('\n', '************************************************************************')
        print(f'Gurobi {model_type} Output Bounds: \n')
        for idx in range(output_dim):
            print(f'{lb[idx]} <= f_{idx}(x) <= {ub[idx]}')
        print('************************************************************************', '\n')

    else: 
        # Getting the nonzero indices
        non_zero_indices = np.nonzero(c)

        # get the lower bound
        m.setObjective(c.T@final_layer_output, gp.GRB.MINIMIZE)
        m.optimize()
        lb = m.ObjVal

        # get the upper bound
        m.setObjective(c.T@final_layer_output, gp.GRB.MAXIMIZE)
        m.optimize()
        ub = m.ObjVal

        property = ""
        for non_zero_idx in non_zero_indices[0]:
            sign = "+" if c[non_zero_idx]>0 else "-"
            property += f"{sign} {np.abs(c[non_zero_idx])}f_{non_zero_idx}(x) "

        # This is to print the bounds on the property
        print('\n', '************************************************************************')
        print(f'Gurobi {model_type} Output Bounds: \n')
        print(f'{lb} <= {property} <= {ub}')
        print('************************************************************************', '\n')
            


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
