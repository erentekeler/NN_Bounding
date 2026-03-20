import torch
from torch import nn
import onnx


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        # self.NN = nn.Sequential(
        #     nn.Linear(20, 15),
        #     nn.ReLU(),
        #     nn.Linear(15, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 3),
        #     nn.ReLU(),
        #     nn.Linear(3, 3)
        # )

        self.NN = nn.Sequential(
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20)
        )

        # # Auto lirpa paper example
        # self.NN = nn.Sequential(
        #     nn.Linear(2, 2),
        #     nn.ReLU(),
        #     nn.Linear(2, 2),
        #     nn.ReLU(),
        #     nn.Linear(2, 1)
        # )
        

    def forward(self, x):
        return self.NN(x)



if __name__=="__main__":
    model = NeuralNetwork()
    input = torch.ones(model.NN[0].weight.shape[1])
    print("Model input: ", input)
    print("Model output: ", model(input))
    print()

    print("NN architecture:")
    for layer in model.NN:
        print(layer)

