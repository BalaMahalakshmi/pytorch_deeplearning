import torch
from torch import nn
class LinearRegressionModel(nn.Module):
    def __init__(self):
        # weights=0.7
        # bias=0.3
        super().__init__()
        self.weights = nn.parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
    def forward(self, x: torch.Tensor):
        print(self.weights * self.bias)