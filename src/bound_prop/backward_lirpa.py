import torch
from torch import nn
from typing import Dict

from src.NN_model import NeuralNetwork
from src.Bounding import Bounding
import numpy as np

# TODO add forward-backward

class backward_lirpa():
    def __init__(self, model, input_range=None,
                eps=None, x_0=None, norm=None, c=None,
                ub_relaxations: Dict =None,
                lb_relaxations: Dict =None):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"

        bound_computer = Bounding(model, input_range=input_range, eps=eps, x_0=x_0, norm=norm, c=c)
        self.layer_information = bound_computer.layer_information 
        self.set_relaxations(ub_relaxations, lb_relaxations) # This function is called to set user defined relaxations, if any given

        self.model = model
        self.c = c.to(self.device) if c is not None else c # Specification vector

        # put all input related parameters in a dictionary, this is later processed to find the dual norm and simplify concretization
        self.input_specs = {"input_range": input_range, "eps": eps, "x_0": x_0, "norm": norm}
        self.process_input_specs()

        
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
        A_lb = self.c.T@torch.eye(A_init_dim, device=self.device) if self.c is not None else torch.eye(A_init_dim, device=self.device)
        A_ub = self.c.T@torch.eye(A_init_dim, device=self.device) if self.c is not None else torch.eye(A_init_dim, device=self.device)

        d_ub = self.c.T@torch.zeros(A_init_dim, device=self.device) if self.c is not None else torch.zeros(A_init_dim, device=self.device)
        d_lb = self.c.T@torch.zeros(A_init_dim, device=self.device) if self.c is not None else torch.zeros(A_init_dim, device=self.device)

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


    def process_input_specs(self):
        '''Here I process the input specs to simplify the concretization.''' 
        if self.input_specs["input_range"] is not None:
            self.input_specs["norm"] = torch.inf
            self.input_specs["dual_norm"] = 1
        else:
            if self.input_specs["norm"] == torch.inf:
                self.input_specs["dual_norm"] = 1
            else:
                self.input_specs["dual_norm"] = self.input_specs["norm"]/(self.input_specs["norm"]-1) # From 1/p + 1/q = 1

    def compute_bounds(self, print_out_bounds=True):
        '''This method computes the bounds via concretization and print&return them.'''
        # First we compute the relaxation matrix and vectors
        A_ub, A_lb, d_lb, d_ub = self.compute_A_d()

        # if input range is not given, so it is a pure norm ball, then directly apply the dual norm
        if self.input_specs["input_range"] is None:
            # min (a_i^Tx + d) over ||x-x_0||_p <= epsilon    =>   -epsilon||a_i||_q + a_i^T x_0 + d
            # max (a_i^Tx + d) over ||x-x_0||_p <= epsilon    =>   epsilon||a_i||_q + a_i^T x_0 + d, beautiful!

            # Since the scalar case (c is given) is unsqueezed above, we can apply the concretization generically (norm over dim=1).
            lb = -self.input_specs["eps"]*torch.norm(A_lb, p=self.input_specs["dual_norm"], dim=1) + torch.matmul(A_lb, self.input_specs["x_0"]) + d_lb
            ub = self.input_specs["eps"]*torch.norm(A_ub, p=self.input_specs["dual_norm"], dim=1) + torch.matmul(A_ub, self.input_specs["x_0"]) + d_ub
        
        else:
            input_range = self.input_specs["input_range"]
            x_sig = (1/2)*(input_range[:, 1] - input_range[:, 0])
            x_mu = (1/2)*(input_range[:, 1] + input_range[:, 0])

            # elementwise computation of the bounds 
            ub = torch.abs(A_ub)@x_sig + A_ub@x_mu + d_ub
            lb = -torch.abs(A_lb)@x_sig + A_lb@x_mu + d_lb

        if print_out_bounds:
            self.print_backward_results(lb=lb, ub=ub)

        return (lb, ub)

            
    def print_backward_results(self, lb, ub):
        # Print the output nicely :) 
        print('\n', '************************************************************************')
        print('Backward Output Bounds: ')
        if self.c is None:
            for idx, (b_lb, b_ub) in enumerate(zip(lb,ub)):
                print(f'{b_lb} <= f_{idx}(x) <= {b_ub}')
        else: 
            # Getting the nonzero indices
            f_cpu_c = self.c.flatten().detach().cpu().numpy() # flattened c vector
            non_zero_indices = np.nonzero(f_cpu_c)

            property = ""
            for non_zero_idx in non_zero_indices[0]:
                sign = "+" if f_cpu_c[non_zero_idx]>0 else "-"
                property += f"{sign} {np.abs(f_cpu_c[non_zero_idx])}f_{non_zero_idx}(x) "

            # Getting the output bounds on the property

            # This is to print the bounds on the property
            print('\n', '************************************************************************')
            print(f'{lb.item()} <= {property} <= {ub.item()}')
        print('************************************************************************', '\n')



if __name__ == "__main__":
    # Fix the seed and initialize the model
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().NN.to(device)

    output_size = model[-1].weight.shape[0]
    c = torch.zeros((output_size,1)).to(device)
    c[0] = -21
    c[15] = -5

    '''This runs it with elementwise infinity norm ball'''
    # # I determine the input shape based on model parameters to be generic
    # # i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
    input_size = model[0].weight.shape[1]
    input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

    backward_lirpa = backward_lirpa(model=model, input_range=input, c=None)
    backward_lb, backward_ub = backward_lirpa.compute_bounds(print_out_bounds=True)


    '''This runs it with a pure norm ball'''
    # # I determine the input shape based on model parameters to be generic
    # input_size = model[0].weight.shape[1]
    # x_0 = torch.ones(input_size, dtype=torch.float32, device=device)
    # norm = 2
    # eps = 10

    # backward_lirpa = backward_lirpa(model=model, eps=eps, x_0=x_0, norm=norm, c=c)
    # backward_lb, backward_ub = backward_lirpa.compute_bounds(print_out_bounds=True)




