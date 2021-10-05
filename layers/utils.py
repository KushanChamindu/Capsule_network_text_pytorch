import torch
from torch import nn

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        # result = torch.mul(tensors[0],tensors[1])
        result = torch.ones(tensors[0].size())
        for t in tensors:
            result *= t
        return result

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1,self.shape[0],self.shape[1])