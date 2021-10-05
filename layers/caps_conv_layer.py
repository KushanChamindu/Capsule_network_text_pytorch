import torch
from torch import nn
from layers.utils import Reshape

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