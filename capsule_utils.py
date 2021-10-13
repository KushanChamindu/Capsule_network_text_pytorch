import torch
from torch.autograd import Variable

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    squre = torch.square(vectors)
    s_squared_norm = torch.sum(squre, axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / torch.sqrt(s_squared_norm + 1e-07)
    output = scale * vectors
    return output

def squash_fn(input, dim=-1):
    norm = input.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * input
def squash_d(x, axis=-1):
    s_squared_norm = torch.sum(torch.square(x), axis, keepdims=True)
    scale = torch.sqrt(s_squared_norm +1e-07)
    return x / scale