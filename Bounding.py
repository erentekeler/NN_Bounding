import torch 
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from NN_model import NeuralNetwork
from IBP import IBP




class Bounding(IBP):
    def __init__(self, model, input):
        super().__init__(model, input)  
        self.compute_relaxations()



    def compute_relaxations(self):
        ''' 
        This function computes the slope and the bias of the upper convex relaxations.

        By default it sets the lower bound of ReLUs as 0.
        
        '''

        # Mask the dataframe and get the activation functions
        mask_activation = self.layer_information["Layer_type"] != "nn.Linear"

        # Getting the activation functions
        activation_functions = self.layer_information.loc[mask_activation]

        # Checking if the NN has the ReLU function, if yes, the relaxations are computed
        # TODO do it for the other activation functions
        if "nn.ReLU" in activation_functions["Layer_type"].unique():
            self.ReLU_upper()
            self.ReLU_lower()


    def ReLU_upper(self):
        '''
        This function computes the upper convex relaxation of the ReLU activation functions
        '''

        def compute_slope(pre_activation_bounds_x):
            pre_activation_bounds_y = torch.relu(pre_activation_bounds_x) # Getting the y values

            # slope = (ReLU(xu) - ReLU(xl)) / (xu-xl) 
            slope = (pre_activation_bounds_y[:, 1] - pre_activation_bounds_y[:, 0])/(pre_activation_bounds_x[:, 1] - pre_activation_bounds_x[:, 0])

            return slope

        def compute_intercept(pre_activation_bounds_x):
            pre_activation_bounds_y = torch.relu(pre_activation_bounds_x) # Getting the y values

            # intercept = -(ReLU(xu) - ReLU(xl))*xl / (xu-xl)
            intercept = -((pre_activation_bounds_y[:, 1]-pre_activation_bounds_y[:, 0])*pre_activation_bounds_x[:, 0])/(pre_activation_bounds_x[:, 1] - pre_activation_bounds_x[:, 0])

            return intercept

        # Masking the ReLU layers 
        ReLU_mask = (self.layer_information["Layer_type"] == "nn.ReLU")

        self.layer_information.loc[ReLU_mask, "Upper_bound_slope"] = self.layer_information.loc[ReLU_mask, "Layer_input"].apply(compute_slope)
        self.layer_information.loc[ReLU_mask, "Upper_bound_bias"] = self.layer_information.loc[ReLU_mask, "Layer_input"].apply(compute_intercept)


    def ReLU_lower(self):
        '''
        This function computes the lower convex relaxation of the ReLU activation functions
        By default the lower bound is y=0
        '''

        def set_zero(pre_activation_bounds_x):
            return torch.zeros(pre_activation_bounds_x.shape[0], 1)

        # Masking the ReLU layers 
        ReLU_mask = (self.layer_information["Layer_type"] == "nn.ReLU")

        self.layer_information.loc[ReLU_mask, "Lower_bound_slope"] = self.layer_information.loc[ReLU_mask, "Layer_input"].apply(set_zero)
        self.layer_information.loc[ReLU_mask, "Lower_bound_bias"] = self.layer_information.loc[ReLU_mask, "Layer_input"].apply(set_zero)


    def plot_relaxations(self, layer_idx, neuron_idx):
        '''
        This function plots the upper and lower convex relaxations for debugging purposes
        '''

        # Get the activation function type TODO implement it for other activation functions too
        activation_function_type = self.layer_information.loc[layer_idx, "Layer_type"]

        activation_function = torch.relu if activation_function_type=="nn.ReLU" else None

        # Getting the intended pre-activation bounds 
        pre_activation_bounds_x = self.layer_information.loc[layer_idx, "Layer_input"][neuron_idx, :]
        pre_activation_bounds_y = activation_function(pre_activation_bounds_x)

        # Plotting pre_activation bounds
        plt.scatter(pre_activation_bounds_x[0], pre_activation_bounds_y[0], s=50, label='$x_{l}$')
        plt.scatter(pre_activation_bounds_x[1], pre_activation_bounds_y[1], s=50, label='$x_{u}$')

        # Plotting the ReLU function
        x_range = np.linspace(pre_activation_bounds_x[0]-0.1, pre_activation_bounds_x[1]+0.1, 200)
        plt.plot(x_range, activation_function(torch.tensor(x_range)), c='black', label="ReLU")

        # Getting the relaxation parameters
        upper_slope = self.layer_information.loc[layer_idx, "Upper_bound_slope"][neuron_idx]
        upper_bias = self.layer_information.loc[layer_idx, "Upper_bound_bias"][neuron_idx]

        lower_slope = self.layer_information.loc[layer_idx, "Lower_bound_slope"][neuron_idx]
        lower_bias = self.layer_information.loc[layer_idx, "Lower_bound_bias"][neuron_idx]

        # Plotting the bounds
        plt.plot(x_range, upper_slope*x_range + upper_bias, linestyle='--', label="Upper Bound")
        plt.plot(x_range, lower_slope*x_range + lower_bias, linestyle='--', label="Lower Bound")

        plt.legend()
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    # Fix the seed and initialize the model
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)

    # I determine the input shape based on model parameters to be generic
    # i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
    input_size = model.NN[0].weight.shape[1]
    input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

    Bounding = Bounding(model, input)
    Bounding.plot_relaxations(1,1)
    Bounding.plot_relaxations(1,2)
    Bounding.plot_relaxations(3,1)