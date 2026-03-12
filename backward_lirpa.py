import torch
from torch import nn
import pandas as pd
import numpy as np
from typing import Dict

from NN_model import NeuralNetwork
from IBP import IBP
from Bounding import Bounding



class backward_lirpa():
    def __init__(self, model, input_range=None,
                eps=None, x_0=None, norm=None, c=None,
                ub_relaxations: Dict =None,
                lb_relaxations: Dict =None):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"

        bound_computer = Bounding(model, input_range)
        self.layer_information = bound_computer.layer_information 
        self.set_relaxations(ub_relaxations, lb_relaxations) # This function is called to set user defined relaxations

        self.model = model
        self.c = c.to(self.device) if c is not None else c # Specification vector

        
    def set_relaxations(self, ub_relaxations, lb_relaxations):
        '''If user passes specific relaxation parameters rather than IBP precomputed bounds, the values are set here'''
        if ub_relaxations is not None:
            for key, value in ub_relaxations.items():
                for column, params in value.items():
                    self.layer_information.at[key, column] = params # Setting the user defined parameters for the relaxation

        if lb_relaxations is not None:
            for key, value in lb_relaxations.items():
                for column, params in value.items():
                    self.layer_information.at[key, column] = params # Setting the user defined parameters for the relaxation


    def out_features(self):
        '''This function gets the number of output features.
           It is to handle the scenarios when the last layer is not linear.'''
        # I iterate backwards to get the last linear layer in the NN
        for layer_idx, layer in self.layer_information[::-1].iterrows():
            if layer["Layer_type"] == "nn.Linear":
                return self.model[layer_idx].weight.shape[0] # Getting the output shape of the NN


    def compute_A_d(self):
        '''In this function the backward lirpa is performed and the A matrix along with the d vector is returned for the upper and lower bounds'''
        # Getting the number of output features
        A_init_dim = self.out_features()

        # initializing the backward mode A matrices and d vectors, blending it with the specification vector c in the beginning
        A_lb = self.c@torch.eye(A_init_dim, device=self.device) if self.c is not None else torch.eye(A_init_dim, device=self.device)
        A_ub = self.c@torch.eye(A_init_dim, device=self.device) if self.c is not None else torch.eye(A_init_dim, device=self.device)

        d_ub = self.c@torch.zeros(A_init_dim, device=self.device) if self.c is not None else torch.zeros(A_init_dim, device=self.device)
        d_lb = self.c@torch.zeros(A_init_dim, device=self.device) if self.c is not None else torch.zeros(A_init_dim, device=self.device)

        for layer_idx, layer in self.layer_information[::-1].iterrows():
            if layer["Layer_type"] == "nn.Linear":
                # Bias vector
                d_ub += A_ub@self.model[layer_idx].bias
                d_lb += A_lb@self.model[layer_idx].bias

                # Slope matrix
                A_ub = A_ub@self.model[layer_idx].weight
                A_lb = A_lb@self.model[layer_idx].weight


            elif layer["Layer_type"] == "nn.ReLU":
                # Getting the slopes and biases of the convex relaxation of the activation function and diagonalizing them
                slope_upper = torch.diag(self.layer_information.loc[layer_idx, "Upper_bound_slope"]).to(self.device)
                bias_upper = self.layer_information.loc[layer_idx, "Upper_bound_bias"].to(self.device)

                slope_lower = torch.diag(self.layer_information.loc[layer_idx, "Lower_bound_slope"]).to(self.device)
                bias_lower = self.layer_information.loc[layer_idx, "Lower_bound_bias"].to(self.device)

                zero_A = torch.zeros_like(A_ub) # Creating the comparison matrix to get the positive and negative elements of A

                # Bias vector
                d_ub += torch.max(zero_A, A_ub)@bias_upper + torch.min(zero_A, A_ub)@bias_lower
                d_lb += torch.max(zero_A, A_lb)@bias_lower + torch.min(zero_A, A_lb)@bias_upper

                # Slope matrix
                A_ub = torch.max(zero_A, A_ub)@slope_upper + torch.min(zero_A, A_ub)@slope_lower
                A_lb = torch.max(zero_A, A_lb)@slope_lower + torch.min(zero_A, A_lb)@slope_upper

        return (A_ub, A_lb, d_lb, d_ub)    


    def compute_bounds(print=False):
        '''Here I compute the bounds via concretization'''
        pass
    
            


if __name__ == "__main__":
    # Fix the seed and initialize the model
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().NN.to(device)

    # I determine the input shape based on model parameters to be generic
    # i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
    input_size = model[0].weight.shape[1]
    input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

    c = torch.zeros(20)
    c[0] = 1

    backward_lirpa = backward_lirpa(model=model, input_range=input, c=None)
    A_ub, A_lb, d_lb, d_ub = backward_lirpa.compute_A_d()
    # backward_lb, backward_ub = backward_lirpa.compute_bounds(print=True)

    print(A_ub)




    # # Auto_lirpa paper example
    # # Fix the seed and initialize the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = NeuralNetwork().NN.to(device)
    # model[0].weight = nn.Parameter(torch.tensor([[2, 1], [-3, 4]], dtype=torch.float32))
    # model[0].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

    # model[2].weight = nn.Parameter(torch.tensor([[4, -2], [2, 1]], dtype=torch.float32))
    # model[2].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

    # model[4].weight = nn.Parameter(torch.tensor([[-2, 1]], dtype=torch.float32))
    # model[4].bias = nn.Parameter(torch.tensor([0], dtype=torch.float32))

    # model = model.to(device)

    # input = torch.tensor([[-2, 2], [-1, 3]]).to(device)

    # ub_relaxations = {1:{"Upper_bound_slope": torch.tensor([0.58, 0.64]),
    #                      "Upper_bound_bias": torch.tensor([2.92, 6.43])},
    #                   3:{"Upper_bound_slope": torch.tensor([0.4375, 1]),
    #                      "Upper_bound_bias": torch.tensor([15.75, 0]),
    #                      "Lower_bound_slope": torch.tensor([0, 1.0])}}

    # backward_lirpa = backward_lirpa(model=model, input_range=input, ub_relaxations=ub_relaxations).compute_A()



    