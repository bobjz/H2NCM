import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchvision import transforms, utils
from torchvision.ops import MLP


class DAG_RNN(nn.Module):
    def __init__(self, DAG, input_size, output_ind, mlp_size,num_hidden_layers,activation,dropout):
        super(DAG_RNN, self).__init__()
        self.dag=DAG
        self.state_size = len(DAG)
        self.output_ind = output_ind
        self.mlp_inputs = [DAG[i][-1][0] for i in range(self.state_size)]
        mlp_struct=[mlp_size]*num_hidden_layers
        mlp_struct.append(1)
        mlp_list=[MLP(self.mlp_inputs[i],mlp_struct,\
                      activation_layer=activation, inplace=None, dropout=dropout) for i in range(self.state_size)]
        self.f = nn.ModuleList(mlp_list)
        #self.h = MLP(self.state_size,[mlp_size,mlp_size,output_size])
    def forward(self, input, hidden):

        d_hidden=[]
        for i in range(len(self.f)):
            mlp_in=torch.concat([hidden[:,self.dag[i][0]],input[:,self.dag[i][1]]],axis=-1)
            d_hidden.append(self.f[i](mlp_in))
        new_hidden=hidden+torch.concat(d_hidden,axis=-1)
        return new_hidden[:,self.output_ind:self.output_ind+1], new_hidden 
    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.state_size)
    
    
class MNODE_LSTM(nn.Module):
    def __init__(self,DAG,input_size=4,latent_size=5,output_ind=0,\
                 mlp_size=32,num_hidden_layers=2,activation=nn.ReLU,dropout=0):
        super(MNODE_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                  hidden_size=latent_size,\
                  num_layers=2,\
                  batch_first=True)
        self.dag_rnn_cell=DAG_RNN(DAG,input_size,output_ind,mlp_size,num_hidden_layers,\
                              activation=activation,dropout=dropout)
    def forward(self,past,s,x):
        #past is N*L*5
        lstm_out, (h0,_)=self.lstm(past)
        h0=h0[0]
        h0=h0[:,1:]
        hidden=torch.concat([s[:,0],h0],axis=-1)
        pred, hidden = self.dag_rnn_cell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.dag_rnn_cell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
        return pred
    