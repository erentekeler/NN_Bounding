import torch
from torch import nn
import pandas as pd
import numpy as np

from NN_model import NeuralNetwork
from IBP import IBP


class backward_lirpa(IBP):
    def __init__(self, model, input):
        super().__init__(model, input)  
        self.compute_relaxations()