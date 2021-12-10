from layers.utils import load_word_embedding_matrix
import math
from layers.elu_layer import Elu_layer
from layers.conv_layer import ConvLayer
from layers.caps_conv_layer import ConvCapsLayer
from layers.routing import Routing, CapsuleNorm
from config import Config

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

class ExtractionNet(nn.Module):
    def __init__(self, word_embed_dim, output_size, hidden_size, capsule_num, filter_ensemble_size, dropout_ratio, intermediate_size, sentence_length):
        super(ExtractionNet, self).__init__()

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings=torch.from_numpy(load_word_embedding_matrix("./embeddings/fasttext_lankadeepa_gossiplanka_300_5")), freeze=True)

        self.elu_layer = Elu_layer(in_channels=1, out_channels=capsule_num,
                                   num_features=word_embed_dim, filter_ensemble_size=filter_ensemble_size)
        self.conv_layer = ConvLayer(in_channels=1, out_channels=capsule_num, num_features=word_embed_dim,
                                    filter_ensemble_size=filter_ensemble_size, dropout_ratio=dropout_ratio)
        self.caps_conv_layer = ConvCapsLayer(
            in_channels=capsule_num, out_channels=intermediate_size[0]*intermediate_size[1], intermediate_size=intermediate_size, dropout_ratio=dropout_ratio, filter_ensemble_size=(int(sentence_length - filter_ensemble_size + 1), 1))  #int(sentence_length - (filter_ensemble_size//2)*2)
        
        self.routing_1 = Routing(num_capsule=16,dim_capsule=16,input_shape=intermediate_size, routing=True,num_routing=3)
        self.routing_2 = Routing(num_capsule=4,dim_capsule=16,input_shape=(16,16), routing=True,num_routing=3)

        self.capsule_norm = CapsuleNorm()
        # self.init_weight()

    # def init_weight(self):
        # torch.nn.init.xavier_normal_(self.routing_1.W)
        # torch.nn.init.xavier_normal_(self.routing_2.W)
        # torch.nn.init.xavier_normal_(self.routing_3.W)

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        embeddings = torch.unsqueeze(embeddings, 1).float()
        elu_layer = self.elu_layer(embeddings)
        conv_layer = self.conv_layer(elu_layer, embeddings)
        caps_conv_layer = self.caps_conv_layer(conv_layer)
        routing_1 = self.routing_1(caps_conv_layer)
        routing_2 = self.routing_2(routing_1)
        capsule_norm = self.capsule_norm(routing_2)
        # print(capsule_norm.size())
        return(capsule_norm)

# config = Config(
#     pretrain_vec=torch.from_numpy(load_word_embedding_matrix("./embeddings/fasttext_lankadeepa_gossiplanka_300_5")))

# model = ExtractionNet(word_embed_dim=300, output_size=4, hidden_size=128,
#                       capsule_num=16, filter_ensemble_size=3, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)

