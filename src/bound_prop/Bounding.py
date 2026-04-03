import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.NN_model import NeuralNetwork



class Bounding():
    def __init__(self, model, method=None, compute_relaxation_params=False, compute_interm_bounds=False):
        self.model = model
        self.method = method # IBP, forward or backward
        
        # initializes the layer_information dataframe
        self.parse_network(compute_relaxation_params=compute_relaxation_params, compute_interm_bounds=compute_interm_bounds) 


    def parse_network(self, compute_relaxation_params, compute_interm_bounds):
        # To keep the layer information neatly, bounds are added here and filled by bound prop methods
        intermediate_bounds_columns = [f'{self.method}_input_bounds', f'{self.method}_output_bounds'] if compute_interm_bounds else []   
        relaxation_params_columns = [f"{self.method}_ub_slope", f"{self.method}_ub_bias", f"{self.method}_lb_slope", f"{self.method}_lb_bias"] if compute_relaxation_params else []                           
        self.layer_information = pd.DataFrame(columns=['Layer_idx', 'Layer_type']
                                                    + intermediate_bounds_columns
                                                    + relaxation_params_columns)
                                                   
        for layer_idx, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                # The layer information is saved here in a dataframe
                self.layer_information.loc[len(self.layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.Linear"}

            if isinstance(layer, nn.ReLU):
                # The layer information is saved here in a dataframe 
                self.layer_information.loc[len(self.layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.ReLU"}


    def add_interm_bound_columns(self, bound_prop_method):
        '''This function is to add the intermediate bounds for the forward and backward methods.'''
        self.layer_information[f'{bound_prop_method}_input_bounds'] = None
        self.layer_information[f'{bound_prop_method}_input_bounds'] = self.layer_information[f'{bound_prop_method}_input_bounds'].astype(object)

        self.layer_information[f'{bound_prop_method}_output_bounds'] = None
        self.layer_information[f'{bound_prop_method}_output_bounds'] = self.layer_information[f'{bound_prop_method}_output_bounds'].astype(object)


    def compute_relaxations(self, pre_activation_bounds_x, layer_idx):
        ''' 
        This function computes the slope and the bias of the upper convex relaxations.

        By default it sets the lower bound of ReLUs as 0.
        
        '''
        # Getting the activation function type
        activation_function = self.layer_information.at[layer_idx, "Layer_type"]

        # if layer is a ReLU activation, compute the relaxation parameters for that
        if activation_function =="nn.ReLU":
            self.ReLU_upper(pre_activation_bounds_x, layer_idx)
            self.ReLU_lower(pre_activation_bounds_x, layer_idx)


    def ReLU_upper(self, pre_activation_bounds_x, layer_idx):
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

        self.layer_information.at[layer_idx, f"{self.method}_ub_slope"] = compute_slope(pre_activation_bounds_x)
        self.layer_information.at[layer_idx, f"{self.method}_ub_bias"] = compute_intercept(pre_activation_bounds_x)


    def ReLU_lower(self, pre_activation_bounds_x, layer_idx):
        '''
        This function computes the lower convex relaxation of the ReLU activation functions
        By default the lower bound is y=0
        '''

        def set_zero(pre_activation_bounds_x):
            return torch.zeros(pre_activation_bounds_x.shape[0])
        
        def set_one(pre_activation_bounds_x):
            return torch.ones(pre_activation_bounds_x.shape[0])

        self.layer_information.at[layer_idx, f"{self.method}_lb_slope"] = set_zero(pre_activation_bounds_x)
        self.layer_information.at[layer_idx, f"{self.method}_lb_bias"] = set_zero(pre_activation_bounds_x)


    def export_relaxation_params(self):
        '''This function exports the relaxation parameters in a compatible way with the bound prop methods.
            Note: Not for IBP or custom, only for forward and backward bounds.'''
        ub_relaxation, lb_relaxations = {}, {}
        for layer_idx, layer in self.layer_information.iterrows():
            ub_relaxation[layer_idx] = {'custom_ub_slope': layer[f"{self.method}_ub_slope"],
                                        'custom_ub_bias': layer[f"{self.method}_ub_bias"]}
            
            lb_relaxations[layer_idx] = {'custom_lb_slope': layer[f"{self.method}_lb_slope"],
                                         'custom_lb_bias': layer[f"{self.method}_lb_bias"]}
            
        return (ub_relaxation, lb_relaxations)


    def plot_relaxations(self, bound_prop_method, layer_idx, neuron_idx):
        '''
        This function plots the upper and lower convex relaxations for debugging purposes
        '''
        # Get the activation function type TODO implement it for other activation functions too
        activation_function_type = self.layer_information.loc[layer_idx, "Layer_type"]

        activation_function = torch.relu if activation_function_type=="nn.ReLU" else None

        # Getting the intended pre-activation bounds 
        pre_activation_bounds_x = self.layer_information.loc[layer_idx, f"{bound_prop_method}_input_bounds"][neuron_idx, :]
        pre_activation_bounds_y = activation_function(pre_activation_bounds_x)

        # Plotting pre_activation bounds
        plt.scatter(pre_activation_bounds_x[0], pre_activation_bounds_y[0], s=50, label='$x_{l}$')
        plt.scatter(pre_activation_bounds_x[1], pre_activation_bounds_y[1], s=50, label='$x_{u}$')

        # Plotting the ReLU function
        x_range = np.linspace(pre_activation_bounds_x[0]-0.1, pre_activation_bounds_x[1]+0.1, 200)
        plt.plot(x_range, activation_function(torch.tensor(x_range)), c='black', label="ReLU")

        # Getting the relaxation parameters
        upper_slope = self.layer_information.loc[layer_idx, f"{self.method}_ub_slope"][neuron_idx]
        upper_bias = self.layer_information.loc[layer_idx, f"{self.method}_ub_bias"][neuron_idx]

        lower_slope = self.layer_information.loc[layer_idx, f"{self.method}_lb_slope"][neuron_idx]
        lower_bias = self.layer_information.loc[layer_idx, f"{self.method}_lb_bias"][neuron_idx]

        # Plotting the bounds
        plt.plot(x_range, upper_slope*x_range + upper_bias, linestyle='--', label="Upper Bound")
        plt.plot(x_range, lower_slope*x_range + lower_bias, linestyle='--', label="Lower Bound")

        plt.legend()
        plt.tight_layout()
        plt.show()