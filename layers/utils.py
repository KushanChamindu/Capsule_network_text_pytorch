import torch
from torch import nn
import numpy as np
import pickle

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


def load_word_embedding_matrix(embedding_matrix_path):
    f = open(embedding_matrix_path, 'rb')
    embedding_matrix = np.array(pickle.load(f))
    return embedding_matrix

def text_preprocessing(data):
    comments = data['comment']
    labels = data['label']

    comments_splitted = []
    for comment in comments:
        lines = []
        try:
            words = comment.split()
            lines += words
        except ValueError:
            continue

        comments_splitted.append(lines)
    return comments_splitted, labels
# print(torch.from_numpy(load_word_embedding_matrix("./embeddings/fasttext_lankadeepa_gossiplanka_300_5")).size())
