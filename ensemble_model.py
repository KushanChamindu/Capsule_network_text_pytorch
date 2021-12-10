from data_load import get_data_set
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
# import Helper    ## !wget https://raw.githubusercontent.com/Iamsdt/DLProjects/master/utils/Helper.py

class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, modelC):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = torch.mean(torch.stack([out1,out2,out3]), dim=0) 
        return out