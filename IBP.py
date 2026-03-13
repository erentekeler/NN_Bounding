import torch 
from torch import nn
from NN_model import NeuralNetwork
import pandas as pd
import numpy as np



class IBP():
    def __init__(self, model, input_range=None, eps=None, x_0=None, norm=None, c=None):
        # setting the NN and the specification vector as class attributes
        self.model = model
        self.c = c

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


    def IBP_ReLU(self, input_range):
        '''
        Since monotonic, just pass through ReLU
        ''' 
        return torch.relu(input_range)
    

    def get_l_llidx(self):
        '''This function returns the last linear layer index.'''
        # I iterate backwards to get the last linear layer in the NN
        for layer_idx in reversed(range(len(self.model))):
            if isinstance(self.model[layer_idx], nn.Linear):
                return layer_idx # Getting the index of the last linear layer


    def compute_bounds(self, store_linear_layers=True, print_out_bounds=True, print_interm_bounds=True):
        '''
        This function iterates over the layers and computes the layer outputs
        Also keeps track of the layer types to set the relaxations
        
        returns: output bounds, and preactivation bounds for each activation function
        '''
        layer_information = pd.DataFrame(columns=['Layer_idx', 'Layer_type', 'Layer_input_bounds', 'Layer_output_bounds']) # To keep the layer information neatly
        for layer_idx, layer in enumerate(self.model):
            # Here the first layer is handled based on the input set definition 
            if layer_idx==0:
                if self.input_specs["input_range"] is not None:  # elementwise infinity norm ball
                    layer_output_bounds = self.IBP_Linear_ew(layer, self.input_specs["input_range"])
                    layer_information.loc[len(layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.Linear",'Layer_input_bounds': self.input_specs["input_range"].detach().cpu(), 'Layer_output_bounds': layer_output_bounds.detach().cpu()}
                else: 
                    layer_output_bounds = self.IBP_Linear_p_norm(layer)
                    layer_information.loc[len(layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.Linear",'Layer_input_bounds': None, 'Layer_output_bounds': layer_output_bounds.detach().cpu()}

            # The last linear layer is important since the specification vector c is handled here
            elif layer_idx==self.get_l_llidx():
                layer_input_bounds = layer_output_bounds
                layer_output_bounds = self.IBP_Linear_ew(layer, layer_input_bounds, c=self.c) # computing layer output based on the specification
                layer_information.loc[len(layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.Linear",'Layer_input_bounds': layer_input_bounds.detach().cpu(), 'Layer_output_bounds': layer_output_bounds.detach().cpu()}

            else: # If it is not a special case, i.e., not the input or the last linear layer
                if isinstance(layer, nn.Linear):
                    layer_input_bounds = layer_output_bounds
                    layer_output_bounds = self.IBP_Linear_ew(layer, layer_input_bounds)

                    # The layer information is saved here in a dataframe
                    layer_information.loc[len(layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.Linear",'Layer_input_bounds': layer_input_bounds.detach().cpu(), 'Layer_output_bounds': layer_output_bounds.detach().cpu()}

                if isinstance(layer, nn.ReLU):
                    layer_input_bounds = layer_output_bounds
                    layer_output_bounds = self.IBP_ReLU(layer_input_bounds) # Compute the next layers input bounds

                    # The layer information is saved here in a dataframe
                    layer_information.loc[len(layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.ReLU",'Layer_input_bounds': layer_input_bounds.detach().cpu(), 'Layer_output_bounds': layer_output_bounds.detach().cpu()}

        self.layer_information = layer_information

        # Checks if bounds will be printed
        if print_out_bounds:
            self.print_IBP_results(print_interm_bounds=print_interm_bounds)

        return self.layer_information


    def print_IBP_results(self, print_interm_bounds=True):
        # Print the output nicely :) 
        print('\n', '************************************************************************')
        print('IBP Output Bounds: ')
        if self.c is None:
            for idx, bounds in enumerate(self.layer_information.Layer_output_bounds.iloc[-1]):
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
            output_bounds = self.layer_information.Layer_output_bounds.iloc[-1].cpu().numpy().flatten()
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
    c = torch.zeros((output_size, 1)).to(device)
    c[0] = 12
    c[1] = 3
    c[4] = 5
    c[12] = -7
    

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
    eps = 100

    IBP = IBP(model, x_0=x_0, norm=norm, eps=eps, c=c)
    IBP.compute_bounds(print_interm_bounds=False, print_out_bounds=True)




    # # Auto_lirpa paper example
    # # I took the parameters from the paper
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = NeuralNetwork().NN.to(device)
    # model[0].weight = nn.Parameter(torch.tensor([[2, 1], [-3, 4]], dtype=torch.float32))
    # model[0].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

    # model[2].weight = nn.Parameter(torch.tensor([[4, -2], [2, 1]], dtype=torch.float32))
    # model[2].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

    # model[4].weight = nn.Parameter(torch.tensor([[-2, 1]], dtype=torch.float32))
    # model[4].bias = nn.Parameter(torch.tensor([0], dtype=torch.float32))

    # model = model.to(device)

    # input_range = torch.tensor([[-2, 2], [-1, 3]]).to(device)

    # IBP = IBP(model, input_range=input_range)
    # IBP.compute_bounds(print_out_bounds=True, print_interm_bounds=False)

