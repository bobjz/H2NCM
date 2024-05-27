import numpy as np
import itertools
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.ops import MLP



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000,device=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout).to(device)
        position = torch.arange(max_len).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).to(device) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model).to(device)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class Transformer(nn.Module):
    def __init__(self, d_model=8, nhead=4, num_encoder_layers=3, \
                  num_decoder_layers=3, dim_feedforward=64, dropout=0.1,\
                  input_size=3, output_size=1, device=None):
        super().__init__()
        self.trans=nn.Transformer(d_model,nhead,num_encoder_layers,\
                                  num_decoder_layers,dim_feedforward,dropout,\
                                  batch_first=True)
        self.pos_enc=PositionalEncoding(d_model,dropout,device=device)
        self.src_encoder=nn.Linear(input_size,d_model)
        self.tgt_encoder=nn.Linear(output_size,d_model)
        self.decoder=nn.Linear(d_model,output_size)
        
        
    def forward(self,src,tgt,mask=None):
        src=self.src_encoder(src)
        tgt=self.tgt_encoder(tgt)
        src=self.pos_enc(src)
        if (mask!=None):
            out=self.trans(src,tgt,tgt_mask=mask)
        else:
            out=self.trans(src,tgt)
        return self.decoder(out)
