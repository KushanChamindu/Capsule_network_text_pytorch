import math
from layers.elu_layer import Elu_layer
from layers.conv_layer import ConvLayer
from layers.caps_conv_layer import ConvCapsLayer
from layers.routing import Routing, CapsuleNorm


import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter



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
        self.routing_2 = Routing(num_capsule=4,dim_capsule=16,input_shape=(16,16), routing=True,num_routing=3)

        self.capsule_norm = CapsuleNorm()

    def forward(self, x):
        elu_layer = self.elu_layer(x)
        conv_layer = self.conv_layer(elu_layer, x)
        # self.caps_conv_layer.kernal_size = (conv_layer.size()[1], 1)
        caps_conv_layer = self.caps_conv_layer(conv_layer)
        routing_1 = self.routing_1(caps_conv_layer)
        routing_2 = self.routing_2(routing_1)
        capsule_norm = self.capsule_norm(routing_2)
        return(capsule_norm)

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
print(model)
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
