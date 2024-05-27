import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchvision import transforms, utils
from torchvision.ops import MLP


class BB_RNN(nn.Module):
    def __init__(self, input_size, output_size, mlp_size, hidden_size,\
                 activation, dropout, num_hidden_layer):
        super(BB_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        mlp_struct1=[mlp_size]*num_hidden_layer
        mlp_struct1.append(hidden_size)
        mlp_struct2=[mlp_size]*num_hidden_layer
        mlp_struct2.append(output_size)
        self.f = MLP(input_size+hidden_size,mlp_struct1,\
                     activation_layer=activation, inplace=None,dropout=dropout)
        self.h = MLP(input_size+hidden_size,mlp_struct2,\
                     activation_layer=activation, inplace=None,dropout=dropout)
    def forward(self, input, hidden):
        #hidden=[i1,w1,i2,w2,i3,w3
        dh = self.f(torch.concat([input,hidden],axis=-1))
        new_hidden = hidden + dh
        return self.h(torch.concat([new_hidden,input],axis=-1)), new_hidden
    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class BBNODE_LSTM(nn.Module):
    def __init__(self,input_size=2,latent_size=5,activation=nn.ReLU,dropout=0,\
                 output_size=1,mlp_size=32,num_hidden_layer=2):
        super(BBNODE_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                          hidden_size=latent_size,\
                          num_layers=2,\
                          batch_first=True)
        self.bb_rnn_cell=BB_RNN(input_size,output_size,mlp_size,hidden_size=latent_size,\
                                activation=activation,dropout=dropout,num_hidden_layer=num_hidden_layer)

    def forward(self,past,s,x):
        #past is N*L*5
        lstm_out, (h0,_)=self.lstm(past)
        h0=h0[0]
        h0=h0[:,1:]
        hidden=torch.concat([s[:,0],h0],axis=-1)
        pred, hidden = self.bb_rnn_cell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.bb_rnn_cell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
        return pred
