import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchvision import transforms, utils
from torchvision.ops import MLP


class DTDSimCell(nn.Module):
    def __init__(self,  input_size=2, param_size=2, mlp_size=16, latent_size=8, num_hidden_layer=2):
        super(DTDSimCell, self).__init__()
        mlp_struct=[mlp_size]*num_hidden_layer
        mlp_struct.append(param_size)
        self.mlp2=MLP(latent_size,mlp_struct,\
                     activation_layer=nn.ReLU, inplace=None,dropout=0)
        self.A=nn.Parameter(torch.zeros(latent_size,latent_size))
    def forward(self,inputs,hidden):
        #basal_insulin_data,bolus_c_data,bolus_t_data,\
        #intake_c_data,intake_t_data,\
        #hr_data, hr_d_data, time_start_data, time_end_data
        x1=inputs[:,0:1]; x2=inputs[:,1:2]
        
        
        #dynamical states
        Y=hidden[1]
        
        
        #parameters
        Z=hidden[0]
        new_Z=torch.matmul(Z,self.A)
        K=torch.abs(self.mlp2(new_Z))
        
        k1,k2=torch.split(K[:,0:2],1,dim=-1) #glucose params
    
        DY=-Y+k1*x1-k2*x2
        
        new_S=Y+DY
        
        return Y+DY, [new_Z,new_S]
    
class DTD_LSTM(nn.Module):
    def __init__(self,input_size=2,latent_size=8,state_size=1,\
                 output_size=1,mlp_size=16,num_hidden_layer=2):
        super(DTD_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                  hidden_size=latent_size,\
                  num_layers=2,\
                  batch_first=True)
        self.dtdcell=DTDSimCell(mlp_size=mlp_size,latent_size=latent_size,num_hidden_layer=num_hidden_layer)
    def forward(self,past,s,x):
        #past is N*L*5
        lstm_out, (h0,c0)=self.lstm(past)
        hidden=[c0[0],s[:,0]]
        pred, hidden = self.dtdcell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.dtdcell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
        return pred
