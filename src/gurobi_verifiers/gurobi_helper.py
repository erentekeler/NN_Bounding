import torch
import pandas as pd
import gurobipy as gp
import numpy as np

def constrain_ReLU(m, layer, layer_idx, layer_input, model_type):

    '''Defining variables for pre and post activation'''
    n_neurons = layer["Upper_bound_slope"].shape[0] # Getting the number of neurons

    # Creating the preactivation variable and bounding it based on IBP limits
    ReLU_input_var = layer_input

    # Creating the post activation variables for the ReLU layer
    post_activation_var = m.addMVar(n_neurons, lb=0, ub=layer["Layer_output_bounds"].cpu().numpy()[:,1], name=f'z_hat_{layer_idx}')


    if model_type == 'triangular':
        # Defining the masks for different scenarios of the ReLU pre activation bounds
        b_positive_mask = (layer["Layer_input_bounds"][:,0]>=0) & (layer["Layer_input_bounds"][:,1]>0) # if preactivations are both positive, covers being the same
        b_negative_mask = (layer["Layer_input_bounds"][:,0]<0) & (layer["Layer_input_bounds"][:,1]<=0) # if preactivations are both negative, covers being the same
        b_zero_mask = (layer["Layer_input_bounds"][:,0]==0)&(layer["Layer_input_bounds"][:,1]==0) # if both are zero
        unstable_mask = (layer["Layer_input_bounds"][:,0]<0) & (layer["Layer_input_bounds"][:,1]>0) # if lb is negative and the ub is positive


        # Getting the relaxation parameters
        upper_slope = layer["Upper_bound_slope"].detach().cpu().numpy()
        upper_bias = layer["Upper_bound_bias"].detach().cpu().numpy()

        # Both positive
        if b_positive_mask.any():
            m.addConstr(post_activation_var[b_positive_mask] == ReLU_input_var[b_positive_mask])

        # Both negative
        if b_negative_mask.any():
            m.addConstr(post_activation_var[b_negative_mask] == 0)

        # Both zero
        if b_zero_mask.any():
            m.addConstr(post_activation_var[b_zero_mask] == 0)

        # lb negative, ub positive
        if unstable_mask.any():
            m.addConstr(post_activation_var[unstable_mask] >= 0) # lower bound y>=0
            m.addConstr(post_activation_var[unstable_mask] >= ReLU_input_var[unstable_mask]) # lower bound y>=x
            m.addConstr(post_activation_var[unstable_mask] <= upper_slope[unstable_mask]*ReLU_input_var[unstable_mask] + upper_bias[unstable_mask]) # upper bound, IBP computed

   
   
    elif model_type == "MILP":
        # Defining the masks for different scenarios of the ReLU pre activation bounds
        b_positive_mask = (layer["Layer_input_bounds"][:,0]>=0) & (layer["Layer_input_bounds"][:,1]>0) # if preactivations are both positive, covers being the same
        b_negative_mask = (layer["Layer_input_bounds"][:,0]<0) & (layer["Layer_input_bounds"][:,1]<=0) # if preactivations are both negative, covers being the same
        b_zero_mask = (layer["Layer_input_bounds"][:,0]==0)&(layer["Layer_input_bounds"][:,1]==0) # if both are zero
        unstable_mask = (layer["Layer_input_bounds"][:,0]<0) & (layer["Layer_input_bounds"][:,1]>0) # if lb is negative and the ub is positive

        # Both positive
        if b_positive_mask.any():
            m.addConstr(post_activation_var[b_positive_mask] == ReLU_input_var[b_positive_mask])

        # Both negative
        if b_negative_mask.any():
            m.addConstr(post_activation_var[b_negative_mask] == 0)

        # Both zero
        if b_zero_mask.any():
            m.addConstr(post_activation_var[b_zero_mask] == 0)

        # lb negative, ub positive
        if unstable_mask.any():
            # getting the number of unstable ReLUs, it is not to create unnecessary binary variables
            unstable_ReLU_count = unstable_mask.sum()
            pre_activation_lb = layer["Layer_input_bounds"][:,0].detach().cpu().numpy()
            pre_activation_ub = layer["Layer_input_bounds"][:,1].detach().cpu().numpy()

            # creating the binary decision variables for the layer
            t = m.addMVar(unstable_ReLU_count, vtype=gp.GRB.BINARY, name=f"t_{layer_idx}")

            # adding the constraints for ReLU
            m.addConstr(post_activation_var[unstable_mask] >= 0) # y>=0
            m.addConstr(post_activation_var[unstable_mask] >= ReLU_input_var[unstable_mask]) # y>=x
            m.addConstr(post_activation_var[unstable_mask] <= ReLU_input_var[unstable_mask] - pre_activation_lb[unstable_mask]*(1-t)) # y<=x-L(1-t)
            m.addConstr(post_activation_var[unstable_mask] <= pre_activation_ub[unstable_mask]*t) # y<=Ut

    return post_activation_var