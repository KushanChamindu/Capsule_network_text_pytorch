import math

from pandas.core.base import SelectionMixin

import torch
from torch import nn
from torch.nn.parameter import Parameter
from capsule_utils import squash
from torch.autograd import Variable
import torch.nn.functional as F


class CapsuleNorm(nn.Module):
    """
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def __init__(self):
        super(CapsuleNorm, self).__init__()

    def forward(self, inputs, **kwargs):
        return torch.sqrt(torch.sum(torch.square(inputs), -1) + 1e-07)

    # def compute_output_shape(self, input_shape):
    #     return input_shape[:-1]

    # def get_config(self):
    #     config = super(Length, self).get_config()
    #     return config


class Routing(nn.Module):

    def __init__(self, num_capsule,
                 dim_capsule,
                 input_shape,
                 routing=True,
                 num_routing=3,
                 l2_constant=0.0001,
                 kernel_initializer='glorot_uniform', **kwargs):

        super(Routing, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routing = routing
        self.num_routing = num_routing
        self.l2_constant = l2_constant
        # self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_num_capsule = input_shape[0]
        self.input_dim_capsule = input_shape[1]

        weights = torch.Tensor(self.input_num_capsule, self.num_capsule,
                                       self.input_dim_capsule, self.dim_capsule)
        
        self.W = Parameter(weights,requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.
        torch.nn.init.xavier_normal_(self.W)
        self.W.data.uniform_(-1,1)

        # Transform matrix
        # self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
        #                                 self.input_dim_capsule, self.dim_capsule],
        #                          initializer=self.kernel_initializer,
        #                          regularizer=l2(self.l2_constant),
        #                          name='capsule_weight',
        #                          trainable=True)

        # self.W = Variable(torch.randn(self.input_num_capsule, self.num_capsule,
        #                               self.input_dim_capsule, self.dim_capsule).type(float), requires_grad=True)

        # initialize weights
        # torch.nn.init.xavier_uniform_(self.W, gain=1.0)

    def forward(self, inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # priors = torch.matmul(self.W.unsqueeze(dim=0), inputs.unsqueeze(dim=1).unsqueeze(dim=-1)).squeeze(dim=-1)
        # inputs_hat = torch.einsum('ijnm,bin->bijm', self.W, inputs)
        # # print("input hat size - ", inputs_hat.size())
        # outputs, _ = self.dynamic_routing(input = inputs_hat,num_iterations=self.num_routing)
        
        # print("Routing started....")
        # print(inputs.shape)
        # print(self.W.shape)
        # inputs_expand = K.expand_dims(inputs, 1)
        # inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # inputs_hat.shape = [None, num_capsule, input_num_capsule, upper capsule length]
        inputs_hat = torch.einsum('ijnm,bin->bijm', self.W, inputs)
        # print("input hat - ",self.W.size())
        # print("input - ", inputs.size())
        # print("weights shape:", self.W.shape)
        # print("inputs shape:", inputs.shape)
        # print('input hat shape:', inputs_hat.shape)
        # dynamic routing
        if self.routing:
            b = Variable(torch.zeros(inputs_hat.size()[0],self.input_num_capsule, self.num_capsule)).to(device)
            
            for i in range(self.num_routing):
                # c shape = [batch_size, num_capsule, input_num_capsule]
                c = F.softmax(b, dim=1)

                # outputs = [batch_size, num_classes, upper capsule length]
                outputs = torch.einsum('bij,bijm->bjm', c, inputs_hat)
                outputs = squash(outputs)
                # print(outputs)
                if i < self.num_routing - 1:
                    # print(b.size())
                    b = b + torch.einsum('bjm,bijm->bij', outputs, inputs_hat)

        # static routing
        else:
            # outputs = [batch_size, num_classes, upper capsule length]
            outputs = torch.sum(inputs_hat, axis=2)
            outputs = squash(outputs)
        # print("outputs shape:", outputs.shape)
        return outputs


    # def compute_output_shape(self, input_shape):
    #     return tuple([None, self.num_capsule, self.dim_capsule])

    # def get_config(self):
    #     config = {
    #         'num_capsule': self.num_capsule,
    #         'dim_capsule': self.dim_capsule,
    #         'routing': self.routing,
    #         'num_routing': self.num_routing,
    #         'l2_constant': self.l2_constant
    #     }
    #     base_config = super(Routing, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


