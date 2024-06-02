import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchvision import transforms, utils
from torchvision.ops import MLP


class UVASimCell(nn.Module):
    def __init__(self,  input_size=4, param_size=28):
        super(UVASimCell, self).__init__()
       
        self.K=nn.Parameter(torch.randn(param_size))
        
    def forward(self,inputs,hidden):
        #basal_insulin_data,bolus_c_data,bolus_t_data,\
        #intake_c_data,intake_t_data,\
        #hr_data, hr_d_data, time_start_data, time_end_data
        insulin=inputs[:,0:1]; carb=inputs[:,1:2]
        other=inputs[:,2:]
        
        #dynamical states
        S=hidden[0]
        Gp,Gt=torch.split(S[:,:2],1,dim=-1) #glucose states
        Ip,Il=torch.split(S[:,2:4],1,dim=-1) #insulin states
        X,XL=torch.split(S[:,4:6],1,dim=-1) #intermediary states
        Qsto1,Qsto2,Qgut=torch.split(S[:,6:9],1,dim=-1) #meal absorption states
        
        
        #parameters
        
        K=torch.abs(self.K)
        k1,k2=torch.split(K[0:2],1,dim=-1) #glucose params
        m1,m2,m3,m4=torch.split(K[2:6],1,dim=-1) #insulin params 
        kgri,D,kmin,kmax,kabs,alpha,beta,b,c,D,BW,f=torch.split(K[6:18],1,dim=-1) #ra params
        kp1,kp2,kp3,ki=torch.split(K[18:22],1,dim=-1) #EGP params
        Uii,Vm0,Vmx,Km0,r1,p2u=torch.split(K[22:],1,dim=-1) # U params
        
        EGP=kp1-kp2*Gp-kp3*XL#+xi*XH
        #Ra system
        Qsto=Qsto1+Qsto2        
        kemptQ=kmin+(kmax-kmin/2)*(\
               torch.tanh(alpha*(Qsto-b*D))-\
               torch.tanh(beta*(Qsto-c*D))+2)
        Ra=f*kabs*Qgut/BW
        
        #Utilization system
        Uid=(Vm0+Vmx*X*r1)*Gt/(Km0+torch.abs(Gt))
        
        
        #Meal
        DQsto1=-kgri*Qsto1+D*carb
        DQsto2=-kemptQ*Qsto2+kgri*Qsto1
        DQgut=-kabs*Qgut+kemptQ*Qsto2
      
        #Utilization
        #insulin 
        DIp=-(m2+m4)*Ip+m1*Il+insulin
        DIl=-(m1+m3)*Il+m2*Ip
        DXL=-ki*(XL-Ip)
        DX=-p2u*X+p2u*Ip
        
        #glucose
        DGp=EGP+Ra-Uii-k1*Gp+k2*Gt
        DGt=-Uid+k1*Gp-k2*Gt
        #DGs=-Ts*Gs+Ts*G
        
        new_S=torch.concat([Gp+DGp, Gt+DGt,\
                        Ip+DIp, Il+DIl,\
                        X+DX, XL+DXL,\
                        Qsto1+DQsto1, Qsto2+DQsto2, Qgut+DQgut],axis=-1)

        return Gp+DGp, [new_S]
    
class UVA_LSTM(nn.Module):
    def __init__(self,input_size=4,latent_size=9,state_size=9,\
                 output_size=1):
        super(UVA_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                  hidden_size=latent_size,\
                  num_layers=2,\
                  batch_first=True)
        self.uvacell=UVASimCell()
    def forward(self,past,s,x):
        #past is N*L*5
        lstm_out, (h0,c0)=self.lstm(past)
        hidden=[torch.concat([s[:,0],h0[0]],axis=-1)]
        pred, hidden = self.uvacell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.uvacell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
        return pred
