import math

import torch
import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn
from torch.nn.parameter import Parameter



class Elu_layer(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, filter_ensemble_size, num_filters):
        super(Elu_layer, self).__init__()
        self.elu_layer_conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               # num_features = self.embedding_size
                               kernel_size=(
                                   filter_ensemble_size, num_features),
                               stride=1, bias=False,
                               )
        # self.elu_layer_conv = nn.ModuleList([
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(
        #                             filter_ensemble_size, num_features), stride=1,bias=False, padding=0)
        #     for _ in range(num_capsules)])
        self.batch_normalization_layer = nn.BatchNorm2d(
        num_filters, eps=1e-05, momentum=0.1, affine=True)

        self.elu_layer = nn.ELU(alpha=1.0, inplace=False)

    def forward(self, x):
        # elu_layer_conv = [capsule(x) for capsule in self.elu_layer_conv]
        # u = torch.stack(u, dim=1)
        # elu_layer_conv = u.view(x.size(0), self.num_routes, -1)
        print(x.type())
        elu_layer_conv = self.elu_layer_conv(x)
        print(elu_layer_conv.size)
        print(elu_layer_conv)
        batch_normalized = self.batch_normalization_layer(elu_layer_conv)
        print(batch_normalized)
        return self.elu_layer(batch_normalized)  # F.elu(batch_normalized, alpha=1.0, inplace=False)  
class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()

  def forward(self, tensors):
    result = torch.ones(tensors[0].size())
    for t in tensors:
      result *= t
    return t
class ConvLayer(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, filter_ensemble_size):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                # num_features = self.embedding_size
                                kernel_size=(
                                    filter_ensemble_size, num_features),
                                stride=1, bias=False,
                                )
        # self.conv_layer = nn.ModuleList([
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(
        #                             filter_ensemble_size, num_features), stride=1,bias=False, padding=0)
        #     for _ in range(num_capsules)])
        self.batch_normalization_layer = nn.BatchNorm1d(
        num_features, eps=1e-05, momentum=0.1, affine=True)
        # self.multiply_layer = Multiply()
        # self.dropout = nn.Dropout(p=0.5, inplace=False)
        

    def forward(self, x):
        conv_layer = self.conv_layer(x)
        # conv_layer = [capsule(x) for capsule in self.conv_layer]
        # u = torch.stack(u, dim=1)
        # conv_layer = u.view(x.size(0), self.num_routes, -1)
        batch_normalized = self.batch_normalization_layer(conv_layer)
        return batch_normalized
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
class ConvCapsLayer(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, filter_ensemble_size,dropout_ratio):
        super(ConvCapsLayer, self).__init__()

        self.h_i = nn.Conv2d(in_channels=128 * 8,
                         out_channels=out_channels,
                         # num_features = self.embedding_size
                         kernel_size=(filter_ensemble_size, 1),
                         stride=1, bias=False,)
        self.batch_normalization_layer = nn.BatchNorm1d(
        (128, 8), eps=1e-05, momentum=0.1, affine=True)
        
        self.dropout_capsule = nn.Dropout(p=self.dropout_ratio, inplace=False)


    def forward(self, x):
        h_i = self.h_i(x)
        h_i = Reshape(shape=(128, 8))(h_i)
        normalized_h_i = self.batch_normalization_layer(h_i)
        relu_output = F.relu(normalized_h_i, alpha=1.0, inplace=False)
        return self.dropout_capsule(relu_output)

class ExtractionNet(nn.Module):
    def __init__(self, word_embed_dim, output_size, hidden_size, capsule_num):
        super(ExtractionNet, self).__init__()

        self.elu_layer = Elu_layer(in_channels=1, out_channels= 4, num_features=10, filter_ensemble_size= 3,num_filters=4)
        # self.conv_layer = ConvLayer(in_channels=1, out_channels=1,num_features=1,)

    def forward(self,x):
        elu_layer = self.elu_layer(x)
        print(elu_layer)
model = ExtractionNet(300,30,128,16)
# x= torch.tensor([1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10], dtype=torch.long)
# x = x.view(1,1,3,10)
# x = x.type(torch.long)
x = torch.tensor([[[[ 1,  2,  3,  4,  5,  6,  7,  8, 10, 10],
          [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
          [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]]],
          [[[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
          [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
          [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]]]
          ], dtype=torch.float32)
print(x.size())
# x = torch.rand(1,1,3,10)
model(x)
for i in model.parameters():
    print(i.size())
# for i in list(model.parameters()): print(i.shape)
print(torch.cuda.is_available())
        
    
        