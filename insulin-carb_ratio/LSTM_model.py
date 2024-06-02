import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchvision import transforms, utils
from torchvision.ops import MLP


class LSTM(nn.Module):
    def __init__(self,input_size=4,output_size=1,latent_size=5,\
                 num_layers=2,dropout=0):
        super(LSTM, self).__init__()
        self.lstm_en=nn.LSTM(input_size=input_size+1,\
                             hidden_size=latent_size,\
                             num_layers=num_layers,\
                             batch_first=True,\
                             dropout=dropout)
        self.lstm_de=nn.LSTM(input_size=input_size,\
                             hidden_size=latent_size,\
                             num_layers=num_layers,\
                             batch_first=True,\
                             dropout=dropout)
        self.proj=nn.Linear(latent_size,output_size)

    def forward(self,past,s,x):

        lstm_out1, (h0,c0)=self.lstm_en(past)
        lstm_out2, _=self.lstm_de(x,(h0,c0))
        return self.proj(lstm_out2)


