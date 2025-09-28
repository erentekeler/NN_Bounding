import torch
from torch import nn
import onnx


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        self.NN = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
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
    input = torch.ones(20)
    model = NeuralNetwork()
    print("Model output: ", model(input))

    for name, module in model.named_modules():
        print(name, module)


    # This is for exporting it as an ONNX model
    # torch.onnx.export(model, input, "model.onnx")
    # onnx_model = onnx.load("model.onnx")

    # for node in onnx_model.graph.node:
    #     print(f"Node name: {node.name}, Operation: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")
