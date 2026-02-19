import torch 
from torch import nn
from NN_model import NeuralNetwork
import pandas as pd



class IBP():
    def __init__(self, model, input):
        self.layer_information = self.run_IBP(model, input)
        

    def IBP_Linear(self, layer, input):
        '''
        This function is to compute the linear layer output range
        
        returns: a tensor with two columns, first column being the lower and the second being the upper bound of the linear layer
        '''
        x_sig = (1/2)*(input[:, 1] - input[:, 0])
        x_mu = (1/2)*(input[:, 1] + input[:, 0])

        Wx_sig = torch.matmul(torch.abs(layer.weight), x_sig)
        Wx_mu = torch.matmul(layer.weight, x_mu)

        y_u = Wx_sig + Wx_mu + layer.bias
        y_l = -Wx_sig + Wx_mu + layer.bias
        
        return torch.cat([y_l.unsqueeze(1), y_u.unsqueeze(1)], dim=1)


    def IBP_ReLU(self, input):
        '''
        Since monotonic, just pass through ReLU
        ''' 
        return torch.relu(input)


    def run_IBP(self, model, input, store_linear_layers=True):
        '''
        Essential function, iterates over the layers and computes the layer outputs
        Also keeps track of the layer types to set the relaxations
        
        returns: output bounds, and preactivation bounds for each activation function
        '''
        pre_layer_input = input # this is the input before it is passed to the next layer

        layer_information = pd.DataFrame(columns=['Layer_idx', 'Layer_type', 'Layer_input', 'Layer_output']) # To keep the layer information neatly
        for layer_idx, layer in enumerate(model):
            if isinstance(layer, nn.Linear):
                pre_activation_bounds = pre_layer_input
                pre_layer_input = self.IBP_Linear(layer, pre_activation_bounds)

                # The layer information is saved here in a dataframe
                layer_information.loc[len(layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.Linear",'Layer_input': pre_activation_bounds.detach().cpu(), 'Layer_output': pre_layer_input.detach().cpu()}

            if isinstance(layer, nn.ReLU):
                pre_activation_bounds = pre_layer_input
                pre_layer_input = self.IBP_ReLU(pre_activation_bounds) # Compute the next layers input bounds

                # The layer information is saved here in a dataframe
                layer_information.loc[len(layer_information), :] = {'Layer_idx': layer_idx, 'Layer_type': "nn.ReLU",'Layer_input': pre_activation_bounds.detach().cpu(), 'Layer_output': pre_layer_input.detach().cpu()}

        return layer_information


    def print_IBP_results(self, verbose=True):
        # Print the output nicely :) 
        print('************************************************************************', '\n')

        print('IBP Output Bounds: ', '\n')
        for idx, bounds in enumerate(self.layer_information.Layer_output.iloc[-1]):
            print(f'{bounds[0]} <= f_{idx}(x) <= {bounds[1]}')

        print('\n************************************************************************', '\n')
        
        if verbose:
            for idx, row in self.layer_information.iterrows():
                layer_type = row.Layer_type
                layer_input = row.Layer_input
                layer_output = row.Layer_output

                print(f'Layer {idx}, {layer_type}, input: \n', layer_input, '\n')

                print(f'Layer {idx}, {layer_type}, output: \n', layer_output, '\n')

            print('\n************************************************************************')


if __name__ == "__main__":
    # Fix the seed and initialize the model
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().NN.to(device)

    # I determine the input shape based on model parameters to be generic
    # i_l is drawn from U[0.5), i_u is drawn from U[0.5 1) to ensure the validitiy of bounds
    input_size = model[0].weight.shape[1]
    input = torch.cat([torch.rand(input_size).unsqueeze(1)*0.5, 0.5*torch.rand(input_size).unsqueeze(1) + 0.5], dim=1).to(device)

    IBP = IBP(model, input)
    IBP.print_IBP_results()



    # Auto_lirpa paper example
    # Fix the seed and initialize the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = NeuralNetwork().NN
    # model[0].weight = nn.Parameter(torch.tensor([[2, 1], [-3, 4]], dtype=torch.float32))
    # model[0].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

    # model[2].weight = nn.Parameter(torch.tensor([[4, -2], [2, 1]], dtype=torch.float32))
    # model[2].bias = nn.Parameter(torch.tensor([0,0], dtype=torch.float32))

    # model[4].weight = nn.Parameter(torch.tensor([-2, 1], dtype=torch.float32))
    # model[4].bias = nn.Parameter(torch.tensor([0], dtype=torch.float32))

    # model = model.to(device)

    # input = torch.tensor([[-2, 2], [-1, 3]]).to(device)

    # IBP = IBP(model, input)

    # IBP.print_IBP_results()

