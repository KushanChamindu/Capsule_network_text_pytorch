import math
from routing import Routing

import torch
import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn
from torch.nn.parameter import Parameter


class Elu_layer(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, filter_ensemble_size):
        super(Elu_layer, self).__init__()
        self.elu_layer_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        # num_features = self.embedding_size
                                        kernel_size=(
                                            filter_ensemble_size, num_features),
                                        stride=1, bias=False,
                                        )
        self.batch_normalization_layer = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True)

        self.elu_layer = nn.ELU(alpha=1.0, inplace=False)

    def forward(self, x):
        elu_layer_conv = self.elu_layer_conv(x)
        batch_normalized = self.batch_normalization_layer(elu_layer_conv)
        # F.elu(batch_normalized, alpha=1.0, inplace=False)
        return self.elu_layer(batch_normalized)


class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        # result = torch.mul(tensors[0],tensors[1])
        result = torch.ones(tensors[0].size())
        for t in tensors:
            result *= t
        return result


class ConvLayer(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, filter_ensemble_size, dropout_ratio):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    # num_features = self.embedding_size
                                    kernel_size=(
                                        filter_ensemble_size, num_features),
                                    stride=1, bias=False,
                                    )
        self.batch_normalization_layer = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True)
        self.multiply_layer = Multiply()
        self.dropout = nn.Dropout(p=dropout_ratio, inplace=False)

    def forward(self, elu_layer, x):
        conv_layer = self.conv_layer(x)
        batch_normalized = self.batch_normalization_layer(conv_layer)
        # gate = torch.mul(elu_layer,batch_normalized)
        gate_layer = self.multiply_layer([elu_layer, batch_normalized])
        return self.dropout(gate_layer)
        # return conv_layer


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1,self.shape[0],self.shape[1])


class ConvCapsLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_ratio, intermediate_size, filter_ensemble_size):
        super(ConvCapsLayer, self).__init__()

        self.h_i = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,  # 128 * 8
                             # num_features = self.embedding_size
                             kernel_size=filter_ensemble_size,
                             stride=1, bias=False,)
        self.reshape_layer = Reshape(shape=intermediate_size)
        self.batch_normalization_layer = nn.BatchNorm1d(
            (intermediate_size[0]), eps=1e-05, momentum=0.1, affine=True)
        self.relu_layer = nn.ReLU(inplace=False)
        self.dropout_capsule = nn.Dropout(p=dropout_ratio, inplace=False)

    def forward(self, x):
        h_i = self.h_i(x)
        h_i = self.reshape_layer(h_i)
        normalized_h_i = self.batch_normalization_layer(h_i)
        # relu_output = F.relu(normalized_h_i, alpha=1.0, inplace=False)
        relu_output = self.relu_layer(normalized_h_i)
        return self.dropout_capsule(relu_output)


class ExtractionNet(nn.Module):
    def __init__(self, word_embed_dim, output_size, hidden_size, capsule_num, filter_ensemble_size, dropout_ratio, intermediate_size, sentence_length):
        super(ExtractionNet, self).__init__()

        self.elu_layer = Elu_layer(in_channels=1, out_channels=capsule_num,
                                   num_features=word_embed_dim, filter_ensemble_size=filter_ensemble_size)
        self.conv_layer = ConvLayer(in_channels=1, out_channels=capsule_num, num_features=word_embed_dim,
                                    filter_ensemble_size=filter_ensemble_size, dropout_ratio=dropout_ratio)
        self.caps_conv_layer = ConvCapsLayer(
            in_channels=capsule_num, out_channels=intermediate_size[0]*intermediate_size[1], intermediate_size=intermediate_size, dropout_ratio=dropout_ratio, filter_ensemble_size=(int(sentence_length - (filter_ensemble_size//2)*2), 1))
        
        self.routing_1 = Routing(num_capsule=16,dim_capsule=16,input_shape=intermediate_size, routing=True,num_routing=3)

    def forward(self, x):
        elu_layer = self.elu_layer(x)
        conv_layer = self.conv_layer(elu_layer, x)
        # self.caps_conv_layer.kernal_size = (conv_layer.size()[1], 1)
        caps_conv_layer = self.caps_conv_layer(conv_layer)
        routing_1 = self.routing_1(caps_conv_layer)
        return(routing_1)

# x= torch.tensor([1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10], dtype=torch.long)
# x = x.view(1,1,3,10)
# x = x.type(torch.long)


# model = ExtractionNet(word_embed_dim=10,output_size=100,hidden_size=128,capsule_num=4,filter_ensemble_size=3, dropout_ratio=0.8, intermediate_size=(16, 4), sentence_length=3)
# x = torch.tensor([[[[ 1,  2,  3,  4,  5,  6,  7,  8, 10, 10],
#           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
#           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]]],
#           [[[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
#           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
#           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]]]
#           ], dtype=torch.float32)

model = ExtractionNet(word_embed_dim=300, output_size=4, hidden_size=128,
                      capsule_num=16, filter_ensemble_size=3, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)
x = torch.rand(10, 1, 30, 300)
print(x.size())
output = model(x)
print(output.size())
# print(output)
# for i in model.parameters():
#     print(i.size())
# for i in list(model.parameters()): print(i.shape)
# print(torch.cuda.is_available())
# x1 = torch.tensor([1, 2, 3, 4])
# y = x1.reshape(2, 2)
# print(y.size())
# print(y)
# z = (torch.unsqueeze(y, 1))
# print(z.size())
# print(z)
# x1 = torch.tensor([1,2,3,4]).view(2,2)
# x2 = torch.tensor([1,2,3,4]).view(2,2)

# print(Multiply()([x1,x2]))
