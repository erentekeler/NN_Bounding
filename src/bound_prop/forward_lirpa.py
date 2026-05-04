import torch
from torch import nn
from typing import Dict

from src.NN_model import NeuralNetwork
from src.bound_prop.Bounding import Bounding
from src.bound_prop.IBP import IBP
import numpy as np



class forward_lirpa(Bounding):
    def __init__(self, model, input_range=None,
                eps=None, x_0=None, norm=None, c=None,
                relaxation_method="IBP", # IBP, forward, custom
                compute_interm_bounds=True, # If true, all intermediate bounds are computed and stored. Only for IBP and custom relaxation methods
                ub_relaxations: Dict =None,
                lb_relaxations: Dict =None):

        # Initializing the class attributes
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.c = c.to(self.device) if c is not None else c # Specification vector

        # put all input related parameters in a dictionary, this is later processed to find the dual norm and simplify concretization
        self.input_specs = {"input_range": input_range, "eps": eps, "x_0": x_0, "norm": norm}
        self.process_input_specs()

        '''Three options for getting the relaxation paramaters, IBP, forward and custom.'''
        self.relaxation_method = relaxation_method
        if relaxation_method == "IBP": # If relaxations are IBP based, compute the relaxation parameters
            IBP_temp = IBP(model, input_range=input_range, eps=eps, x_0=x_0, norm=norm, c=c, compute_relaxation_params=True)
            IBP_temp.compute_bounds(print_out_bounds=False, print_interm_bounds=False)
            if compute_interm_bounds: # it is possible to compute the intermediate bounds or directly the output bounds
                IBP_temp.add_interm_bound_columns(bound_prop_method="forward")
            self.layer_information = IBP_temp.layer_information
        
        elif relaxation_method == "forward":
            # initializing the bounding object, layer_information dataframe comes from here
            # If pure forward, then relaxation parameter and intermediate bound computation is needed.
            super().__init__(model=model, method=relaxation_method, compute_relaxation_params=True, compute_interm_bounds=True)  
        
        elif relaxation_method == "custom":
            # initializing the bounding object, layer_information dataframe comes from here 
            super().__init__(model=model, method=relaxation_method, compute_relaxation_params=True)
            if compute_interm_bounds: # it is possible to compute the intermediate bounds or directly the output bound
                self.add_interm_bound_columns(bound_prop_method="forward")
            self.set_relaxations(ub_relaxations, lb_relaxations) # This function is called to set user defined relaxations

        self.compute_interm_bounds = compute_interm_bounds

        
    def set_relaxations(self, ub_relaxations, lb_relaxations):
        '''If user passes specific relaxation parameters, the values are set here'''
        if ub_relaxations is not None:
            for key, value in ub_relaxations.items():
                for column, params in value.items():
                    self.layer_information.at[key, column] = params # Setting the user defined parameters for the relaxation

        if lb_relaxations is not None:
            for key, value in lb_relaxations.items():
                for column, params in value.items():
                    self.layer_information.at[key, column] = params # Setting the user defined parameters for the relaxation


    def in_features(self):
        '''This function gets the number of input features.'''
        return self.model[0].weight.shape[1] # Getting the number of input features of the NN


    def compute_A_d(self, model, layer_information):
        '''In this function the forward lirpa is performed and the A matrix along with the d vector is returned for the upper and lower bounds'''
        # Getting the number of input features
        A_init_dim = self.in_features()

        # initializing the forward mode A matrices and d vectors
        A_lb = torch.eye(A_init_dim, device=self.device)
        A_ub = torch.eye(A_init_dim, device=self.device)

        d_ub = torch.zeros(A_init_dim, device=self.device)
        d_lb = torch.zeros(A_init_dim, device=self.device)

        A_ub_temp, A_lb_temp = None, None # These are to keep the diag(alpha_ub)A_ub and diag(alpha_lb)A_lb terms

        for layer_idx, layer in layer_information.iterrows():
            if layer_idx == 0:
                # Bias vector
                d_ub = model[layer_idx].bias
                d_lb = model[layer_idx].bias

                # A matrix
                A_ub = model[layer_idx].weight@A_ub
                A_lb = model[layer_idx].weight@A_lb
            else:
                if layer["Layer_type"] == "nn.Linear":
                    zero_W = torch.zeros_like(model[layer_idx].weight) # Creating the comparison matrix to get the positive and negative elements of A

                    # A matrix
                    A_ub_temp = A_ub.clone()
                    A_lb_temp = A_lb.clone()
                    A_ub = torch.max(zero_W, model[layer_idx].weight)@A_ub_temp + torch.min(zero_W, model[layer_idx].weight)@A_lb_temp
                    A_lb = torch.max(zero_W, model[layer_idx].weight)@A_lb_temp + torch.min(zero_W, model[layer_idx].weight)@A_ub_temp

                    # Bias vector
                    d_ub_temp = d_ub.clone()
                    d_lb_temp = d_lb.clone()
                    d_ub = torch.max(zero_W, model[layer_idx].weight)@d_ub_temp + torch.min(zero_W, model[layer_idx].weight)@d_lb_temp + model[layer_idx].bias
                    d_lb = torch.max(zero_W, model[layer_idx].weight)@d_lb_temp + torch.min(zero_W, model[layer_idx].weight)@d_ub_temp + model[layer_idx].bias

                elif layer["Layer_type"] == "nn.ReLU":
                    # Getting the slopes and biases of the convex relaxation of the activation function and diagonalizing them
                    slope_upper = torch.diag(layer_information.loc[layer_idx, f"{self.relaxation_method}_ub_slope"]).to(self.device)
                    bias_upper = layer_information.loc[layer_idx, f"{self.relaxation_method}_ub_bias"].to(self.device)

                    slope_lower = torch.diag(layer_information.loc[layer_idx, f"{self.relaxation_method}_lb_slope"]).to(self.device)
                    bias_lower = layer_information.loc[layer_idx, f"{self.relaxation_method}_lb_bias"].to(self.device)

                    # A matrix
                    A_ub = slope_upper@A_ub
                    A_lb = slope_lower@A_lb

                    # Bias vector
                    d_ub = slope_upper@d_ub + bias_upper
                    d_lb = slope_lower@d_lb + bias_lower

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


    def concretize_bounds(self, output_bound_params, c, print_out_bounds):
        '''This method computes the bounds via concretization and print&return them.'''
        # First we compute the relaxation matrix and vectors
        A_ub, A_lb, d_lb, d_ub = output_bound_params # unpacking (A_ub, A_lb, d_lb, d_ub)
        A_ub_temp, A_lb_temp, d_lb_temp, d_ub_temp = A_ub.clone(), A_lb.clone(), d_lb.clone(), d_ub.clone()

        # if input range is not given, so it is a pure norm ball, then directly apply the dual norm
        if self.input_specs["input_range"] is None:
            # min (a_i^Tx + d) over ||x-x_0||_p <= epsilon    =>   -epsilon||a_i||_q + a_i^T x_0 + d
            # max (a_i^Tx + d) over ||x-x_0||_p <= epsilon    =>   epsilon||a_i||_q + a_i^T x_0 + d, beautiful!

            if c is None: # I'll make this part more efficient, my idea is to make c an I matrix if it is given None.
                lb = -self.input_specs["eps"]*torch.norm(A_lb, p=self.input_specs["dual_norm"], dim=1) + torch.matmul(A_lb, self.input_specs["x_0"]) + d_lb
                ub = self.input_specs["eps"]*torch.norm(A_ub, p=self.input_specs["dual_norm"], dim=1) + torch.matmul(A_ub, self.input_specs["x_0"]) + d_ub
            else:
                # getting the diagonal c to compute new A_lb, A_ub, d_lb and d_ub.
                # if c is positive inequality is preserved else flipped
                c_diag = torch.diag(c.flatten())
                c_zeros = torch.zeros_like(c_diag)

                A_lb = torch.max(c_zeros, c_diag)@A_lb_temp + torch.min(c_zeros, c_diag)@A_ub_temp
                A_ub = torch.max(c_zeros, c_diag)@A_ub_temp + torch.min(c_zeros, c_diag)@A_lb_temp

                d_lb = torch.max(c_zeros, c_diag)@d_lb_temp + torch.min(c_zeros, c_diag)@d_ub_temp
                d_ub = torch.max(c_zeros, c_diag)@d_ub_temp + torch.min(c_zeros, c_diag)@d_lb_temp

                sum_vector = torch.ones_like(c)
                lb = -self.input_specs["eps"]*torch.norm(sum_vector.T@A_lb, p=self.input_specs["dual_norm"]) + torch.matmul(sum_vector.T @ A_lb, self.input_specs["x_0"]) + sum_vector.T@d_lb
                ub = self.input_specs["eps"]*torch.norm(sum_vector.T@A_ub, p=self.input_specs["dual_norm"]) + torch.matmul(sum_vector.T @ A_ub, self.input_specs["x_0"]) + sum_vector.T@d_ub
        else:
            input_range = self.input_specs["input_range"]
            x_sig = (1/2)*(input_range[:, 1] - input_range[:, 0])
            x_mu = (1/2)*(input_range[:, 1] + input_range[:, 0])

            if c is None:
                # elementwise computation of the bounds 
                ub = torch.abs(A_ub)@x_sig + A_ub@x_mu + d_ub
                lb = -torch.abs(A_lb)@x_sig + A_lb@x_mu + d_lb
            else:
                # getting the diagonal c to compute new A_lb, A_ub, d_lb and d_ub.
                # if c is positive inequality is preserved else flipped
                c_diag = torch.diag(c.flatten())
                c_zeros = torch.zeros_like(c_diag)
                
                A_lb = torch.max(c_zeros, c_diag)@A_lb_temp + torch.min(c_zeros, c_diag)@A_ub_temp
                A_ub = torch.max(c_zeros, c_diag)@A_ub_temp + torch.min(c_zeros, c_diag)@A_lb_temp

                d_lb = torch.max(c_zeros, c_diag)@d_lb_temp + torch.min(c_zeros, c_diag)@d_ub_temp
                d_ub = torch.max(c_zeros, c_diag)@d_ub_temp + torch.min(c_zeros, c_diag)@d_lb_temp

                sum_vector = torch.ones_like(c)
                ub = torch.abs(sum_vector.T@A_ub)@x_sig + sum_vector.T@A_ub@x_mu + sum_vector.T@d_ub
                lb = -torch.abs(sum_vector.T@A_lb)@x_sig + sum_vector.T@A_lb@x_mu + sum_vector.T@d_lb

        if print_out_bounds:
            self.print_forward_results(lb=lb, ub=ub)

        return torch.cat([lb.unsqueeze(1), ub.unsqueeze(1)], dim=1)
    

    def compute_intermediate_bounds(self, print_out_bounds):
        '''This function computes the bounds on the partitions of the NN.
           By doing so layer by layer, intermediate bounds are computed.
           If relaxation mode is forward, it also computes forward mode based relaxation parameters.'''
        layer_output_bounds = None 
        for layer_idx, layer in enumerate(self.model): # going over the layers
            is_last = (layer_idx == len(self.model)-1) # check for the last layer
            print_bounds = print_out_bounds and is_last # if it is the last layer, then print the bounds
            if isinstance(layer, nn.Linear): 
                # Assigning the layer input bounds and saving them into the layer information df
                layer_input_bounds = layer_output_bounds
                self.layer_information.at[layer_idx, 'forward_input_bounds'] = layer_input_bounds.detach().cpu() if layer_input_bounds is not None else None

                # Get the NN from the input to the current layer
                model = self.model[:layer_idx+1] 
                layer_information = self.layer_information[:layer_idx+1]
                
                # Then compute the output bound params and concretize bounds up until the current layer
                output_bound_params = self.compute_A_d(model=model, layer_information=layer_information)
                layer_output_bounds = self.concretize_bounds(output_bound_params=output_bound_params, c=self.c if is_last else None, print_out_bounds=print_bounds)
                
                # Storing the bounds in the layer_information df
                self.layer_information.at[layer_idx, 'forward_output_bounds'] = layer_output_bounds.detach().cpu()
            
            elif isinstance(layer, nn.ReLU):
                # Assigning the layer input bounds and saving them into the layer information df
                layer_input_bounds = layer_output_bounds
                self.layer_information.at[layer_idx, 'forward_input_bounds'] = layer_input_bounds.detach().cpu()

                # If relaxation method is forward, then relaxation parameters are computed
                if self.relaxation_method=="forward":    
                    self.compute_relaxations(pre_activation_bounds_x=layer_input_bounds, layer_idx=layer_idx)

                # Get the NN from the input to the current layer
                model = self.model[:layer_idx+1] 
                layer_information = self.layer_information[:layer_idx+1]

                # Then compute the output bound params and concretize bounds up until the current layer
                output_bound_params = self.compute_A_d(model=model, layer_information=layer_information)
                layer_output_bounds = self.concretize_bounds(output_bound_params=output_bound_params, c=self.c if is_last else None, print_out_bounds=print_bounds)
                
                # Storing the bounds in the layer_information df
                self.layer_information.at[layer_idx, 'forward_output_bounds'] = layer_output_bounds.detach().cpu()

        return layer_output_bounds, output_bound_params # last computed layer bound = output bound
            

    def compute_bounds(self, print_out_bounds=True):
        if (self.relaxation_method=="IBP") or (self.relaxation_method=="custom"):
            if self.compute_interm_bounds:
                output_bounds, output_bound_params = self.compute_intermediate_bounds(print_out_bounds=print_out_bounds)
                self.A_ub, self.A_lb, self.d_lb, self.d_ub = output_bound_params # Saving the output bound parameters
                return output_bounds[:, 0], output_bounds[:, 1] 
            else:
                output_bound_params = self.compute_A_d(model=self.model, layer_information=self.layer_information, c=self.c)
                output_bounds = self.concretize_bounds(output_bound_params=output_bound_params, print_out_bounds=print_out_bounds)
                self.A_ub, self.A_lb, self.d_lb, self.d_ub = output_bound_params # Saving the output bound parameters
                return output_bounds[:, 0], output_bounds[:, 1] 
        elif self.relaxation_method == "forward": # must compute the intermediate bounds
            output_bounds, output_bound_params = self.compute_intermediate_bounds(print_out_bounds=print_out_bounds)
            self.A_ub, self.A_lb, self.d_lb, self.d_ub = output_bound_params # Saving the output bound parameters
            return output_bounds[:, 0], output_bounds[:, 1] 
        
    
    def get_output_bound_params(self):
        '''This function returns the output bound parameters'''
        return (self.A_ub, self.A_lb, self.d_lb, self.d_ub)

            
    def print_forward_results(self, lb, ub):
        # Print the output nicely :) 
        print('\n', '************************************************************************')
        print(f'Forward Output Bounds with {self.relaxation_method} relaxations: ')
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
    # c[0] = 1
    # c[15] = 5

    '''This runs it with elementwise infinity norm ball'''
    # I determine the input shape based on model parameters to be generic
    # i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
    input_size = model[0].weight.shape[1]
    input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

    forward_lirpa = forward_lirpa(model=model, input_range=input, c=None, relaxation_method="IBP", compute_interm_bounds=True)
    forward_lb, forward_ub = forward_lirpa.compute_bounds(print_out_bounds=True)


    '''This runs it with a pure norm ball'''
    # # I determine the input shape based on model parameters to be generic
    # input_size = model[0].weight.shape[1]
    # x_0 = torch.ones(input_size, dtype=torch.float32, device=device)
    # norm = 2
    # eps = 10

    # forward_lirpa = forward_lirpa(model=model, eps=eps, x_0=x_0, norm=norm, c=None)
    # forward_lb, forward_ub = forward_lirpa.compute_bounds(print_out_bounds=True)



