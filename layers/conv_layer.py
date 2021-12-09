import torch
from torch import nn
from layers.utils import Multiply

class ConvLayer(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, filter_ensemble_size, dropout_ratio):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    # num_features = self.embedding_size
                                    kernel_size=(
                                        filter_ensemble_size, num_features),
                                    stride=1, bias=False
                                    )
        self.batch_normalization_layer = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True)
        self.multiply_layer = Multiply()
        self.dropout = nn.Dropout(p=dropout_ratio, inplace=False)

    def forward(self, elu_layer, x):
        conv_layer = self.conv_layer(x)
        batch_normalized = self.batch_normalization_layer(conv_layer)
        gate_layer = torch.mul(elu_layer,batch_normalized)
        # gate_layer = self.multiply_layer([elu_layer, batch_normalized])
        return self.dropout(gate_layer)
        # return conv_layer