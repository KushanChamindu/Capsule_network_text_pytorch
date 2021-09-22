import torch

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = torch.sum(torch.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / torch.sqrt(s_squared_norm + 1e-07)
    return scale * vectors