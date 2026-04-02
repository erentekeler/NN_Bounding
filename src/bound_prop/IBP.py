import torch 
from torch import nn
import pandas as pd
import numpy as np

from src.NN_model import NeuralNetwork
from src.Bounding import Bounding


class IBP(Bounding):
    def __init__(self, model, input_range=None, eps=None, x_0=None, norm=None, c=None, compute_relaxation_params=False):
        super().__init__(model=model, method="IBP", compute_relaxation_params=compute_relaxation_params) # initializing the bounding object, layer_information dataframe comes from here

        # setting the NN and the specification vector as class attributes
        self.model = model
        self.c = c

        self.compute_relaxation_params = compute_relaxation_params # If true, relaxation parameters are computed

        # put all input related parameters in a dictionary, this is later processed to find the dual norm and simplify concretization
        self.input_specs = {"input_range": input_range, "eps": eps, "x_0": x_0, "norm": norm}
        self.process_input_specs()


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
        

    def IBP_Linear_ew(self, layer, input_range, c=None):
        '''
        This function computes the linear layer output bounds when the input is elementwise infinity norm ball
        
        returns: a tensor with two columns, first column being the lower and the second being the upper bound of the linear layer
        '''
        x_sig = (1/2)*(input_range[:, 1] - input_range[:, 0])
        x_mu = (1/2)*(input_range[:, 1] + input_range[:, 0])

        if c is None:
            Wx_sig = torch.matmul(torch.abs(layer.weight), x_sig)
            Wx_mu = torch.matmul(layer.weight, x_mu)

            y_u = Wx_sig + Wx_mu + layer.bias
            y_l = -Wx_sig + Wx_mu + layer.bias
        else: # In case the specification is given
            Wx_sig = torch.matmul(torch.abs(self.c.T@layer.weight), x_sig)
            Wx_mu = torch.matmul(self.c.T@layer.weight, x_mu)

            y_u = Wx_sig + Wx_mu + self.c.T@layer.bias
            y_l = -Wx_sig + Wx_mu + self.c.T@layer.bias

        return torch.cat([y_l.unsqueeze(1), y_u.unsqueeze(1)], dim=1)
    

    def IBP_Linear_p_norm(self, layer):
        '''
        This function computes the linear layer output bounds when the input is a pure norm ball: ||x-x_0||_p<=epsilon
        
        returns: a tensor with two columns, first column being the lower and the second being the upper bound of the linear layer
        '''
        # min (w_i^Tx + b) over ||x-x_0||_p <= epsilon    =>   -epsilon||w_i||_q + w_i^T x_0 + b
        # max (w_i^Tx + b) over ||x-x_0||_p <= epsilon    =>   epsilon||w_i||_q + w_i^T x_0 + b, beautiful!
        # Getting the layer parameters
        W = layer.weight
        b = layer.bias

        # This is the equivalent of iterating over the natural basis specification vectors 'c'. This is to compute the first layer output bounds
        y_lb = -self.input_specs["eps"]*torch.norm(W, p=self.input_specs["dual_norm"], dim=1) + W@self.input_specs["x_0"] + b
        y_ub = self.input_specs["eps"]*torch.norm(W, p=self.input_specs["dual_norm"], dim=1) + W@self.input_specs["x_0"] + b
        
        return torch.cat([y_lb.unsqueeze(1), y_ub.unsqueeze(1)], dim=1)
    

    def apply_c(self, bounds):
        '''Used only when last layer is ReLU, sign-split on per-neuron bounds.'''
        c_vec = self.c.flatten()
        c_pos = torch.clamp(c_vec, min=0)
        c_neg = torch.clamp(c_vec, max=0)
        lb = c_pos @ bounds[:, 0] + c_neg @ bounds[:, 1] # flip and sum if negative, preserve and sum if positive
        ub = c_pos @ bounds[:, 1] + c_neg @ bounds[:, 0]
        return torch.stack([lb, ub]).unsqueeze(0)


    def IBP_ReLU(self, input_range):
        '''
        Since monotonic, just pass through ReLU
        ''' 
        return torch.relu(input_range)


    def compute_bounds(self, print_out_bounds=True, print_interm_bounds=True):
        '''
        This function iterates over the layers and computes the layer outputs
        Also keeps track of the layer types to set the relaxations
        
        returns: output bounds, and preactivation bounds for each activation function
        '''

        for layer_idx, layer in enumerate(self.model):
            is_last = (layer_idx==len(self.model)-1)
            # Here the first layer is handled based on the input set definition 
            if layer_idx==0:
                if self.input_specs["input_range"] is not None:  # elementwise infinity norm ball
                    layer_output_bounds = self.IBP_Linear_ew(layer, self.input_specs["input_range"])
                    self.layer_information.loc[layer_idx, ['IBP_input_bounds', 'IBP_output_bounds']] = [self.input_specs["input_range"].detach().cpu(), layer_output_bounds.detach().cpu()]
                else: 
                    layer_output_bounds = self.IBP_Linear_p_norm(layer)
                    self.layer_information.loc[layer_idx, ['IBP_input_bounds', 'IBP_output_bounds']] = [None, layer_output_bounds.detach().cpu()]

            else: # If it is not a special case, i.e., not the input or the last linear layer
                if isinstance(layer, nn.Linear):
                    layer_input_bounds = layer_output_bounds
                    c = self.c if (is_last and self.c is not None) else None
                    layer_output_bounds = self.IBP_Linear_ew(layer, layer_input_bounds, c=c)
                    # The layer information is saved here in a dataframe
                    self.layer_information.loc[layer_idx, ['IBP_input_bounds', 'IBP_output_bounds']] = [layer_input_bounds.detach().cpu(), layer_output_bounds.detach().cpu()]

                if isinstance(layer, nn.ReLU):
                    layer_input_bounds = layer_output_bounds
                    layer_output_bounds = self.IBP_ReLU(layer_input_bounds) # Compute the next layers input bounds
                    c_applied_bounds = self.apply_c(layer_output_bounds) if (is_last and self.c is not None) else layer_output_bounds

                    # The layer information is saved here in a dataframe
                    self.layer_information.loc[layer_idx, ['IBP_input_bounds', 'IBP_output_bounds']] = [layer_input_bounds.detach().cpu(), c_applied_bounds.detach().cpu()]

                    '''Computing the relaxation parameters based on the input bounds of the activation function'''
                    if self.compute_relaxation_params: 
                        self.compute_relaxations(layer_input_bounds.detach().cpu(), layer_idx)


        self.layer_information = self.layer_information

        # Checks if bounds will be printed
        if print_out_bounds:
            self.print_IBP_results(print_interm_bounds=print_interm_bounds)

        return self.layer_information


    def print_IBP_results(self, print_interm_bounds=True):
        # Print the output nicely :) 
        print('\n', '************************************************************************')
        print('IBP Output Bounds: ')
        if self.c is None:
            for idx, bounds in enumerate(self.layer_information["IBP_output_bounds"].iloc[-1]):
                print(f'{bounds[0]} <= f_{idx}(x) <= {bounds[1]}')
        else: 
            # Getting the nonzero indices
            f_cpu_c = self.c.flatten().detach().cpu().numpy() # flattened c vector
            non_zero_indices = np.nonzero(f_cpu_c)

            property = ""
            for non_zero_idx in non_zero_indices[0]:
                sign = "+" if f_cpu_c[non_zero_idx]>0 else "-"
                property += f"{sign} {np.abs(f_cpu_c[non_zero_idx])}f_{non_zero_idx}(x) "

            # Getting the output bounds on the property
            output_bounds = self.layer_information["IBP_output_bounds"].iloc[-1].cpu().numpy().flatten()
            lb = output_bounds[0]
            ub = output_bounds[1]

            # This is to print the bounds on the property
            print('\n', '************************************************************************')
            print(f'{lb} <= {property} <= {ub}')
        print('************************************************************************', '\n')

        
        if print_interm_bounds:
            for idx, row in self.layer_information.iterrows():
                layer_type = row.Layer_type
                layer_input_bounds = row.Layer_input_bounds
                layer_output_bounds = row.Layer_output_bounds

                print(f'Layer {idx}, {layer_type}, input: \n', layer_input_bounds, '\n')

                print(f'Layer {idx}, {layer_type}, output: \n', layer_output_bounds, '\n')

            print('\n************************************************************************')


if __name__ == "__main__":
    # Fix the seed and initialize the model
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().NN.to(device)

    output_size = model[-1].weight.shape[0]
    c = torch.zeros((output_size,1)).to(device)
    c[0] = 1
    c[15] = -18
    

    '''This part runs IBP with the elementwise infinity norm ball'''
    # # I determine the input shape based on model parameters to be generic
    # # i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
    # input_size = model[0].weight.shape[1]
    # input_range = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

    # IBP = IBP(model, input_range=input_range, c=c)
    # IBP.compute_bounds(print_interm_bounds=False, print_out_bounds=True)

    '''This runs it with a pure norm ball'''
    # I determine the input shape based on model parameters to be generic
    input_size = model[0].weight.shape[1]
    x_0 = torch.ones(input_size, dtype=torch.float32, device=device)
    norm = 2
    eps = 10

    IBP = IBP(model, x_0=x_0, norm=norm, eps=eps, c=c, compute_relaxation_params=True)
    IBP.compute_bounds(print_interm_bounds=False, print_out_bounds=True)


