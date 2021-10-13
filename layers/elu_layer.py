import torch
from torch import nn

class Elu_layer(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, filter_ensemble_size):
        super(Elu_layer, self).__init__()
        self.elu_layer_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        # num_features = self.embedding_size
                                        kernel_size=(
                                            filter_ensemble_size, num_features),
                                        stride=1, bias=False
                                        )
        self.batch_normalization_layer = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True)

        self.elu_layer = nn.ELU(alpha=1.0, inplace=False)

    def forward(self, x):
        elu_layer_conv = self.elu_layer_conv(x)
        batch_normalized = self.batch_normalization_layer(elu_layer_conv)
        # F.elu(batch_normalized, alpha=1.0, inplace=False)
        return self.elu_layer(batch_normalized)