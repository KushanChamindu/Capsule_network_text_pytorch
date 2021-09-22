import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
from capsule_utils import squash


class CapsuleNorm(nn.Module):
    """
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """

    def forward(self, inputs, **kwargs):
        return torch.sqrt(torch.sum(torch.square(inputs), -1) + 1e-07)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    # def get_config(self):
    #     config = super(Length, self).get_config()
    #     return config

class Routing(nn.Module):

    def __init__(self, num_capsule,
                 dim_capsule,
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

    def build(self, input_shape):

        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
                                        self.input_dim_capsule, self.dim_capsule],
                                 initializer=self.kernel_initializer,
                                 regularizer=l2(self.l2_constant),
                                 name='capsule_weight',
                                 trainable=True)
        self.built = True

    def call(self, inputs, training=True):
        # print("Routing started....")
        # print(inputs.shape)
        # print(self.W.shape)
        # inputs_expand = K.expand_dims(inputs, 1)
        # inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # inputs_hat.shape = [None, num_capsule, input_num_capsule, upper capsule length]
        inputs_hat = tf.einsum('ijnm,bin->bijm', self.W, inputs)
        print("weights shape:", self.W.shape)
        print("inputs shape:", inputs.shape)
        print('input hat shape:', inputs_hat.shape)
        # dynamic routing
        if self.routing:
            b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.input_num_capsule, self.num_capsule])

            for i in range(self.num_routing):
                # c shape = [batch_size, num_capsule, input_num_capsule]
                c = tf.nn.softmax(b)

                # outputs = [batch_size, num_classes, upper capsule length]
                outputs = tf.einsum('bij,bijm->bjm', c, inputs_hat)

                outputs = squash(outputs)

                if i < self.routing - 1:
                    b += tf.einsum('bjm,bijm->bij', outputs, inputs_hat)

        # static routing
        else:
            # outputs = [batch_size, num_classes, upper capsule length]
            outputs = K.sum(inputs_hat, axis=2)
            outputs = squash(outputs)
        # print("outputs shape:", outputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routing': self.routing,
            'num_routing': self.num_routing,
            'l2_constant': self.l2_constant
        }
        base_config = super(Routing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
