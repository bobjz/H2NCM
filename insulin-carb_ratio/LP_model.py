import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchvision import transforms, utils
from torchvision.ops import MLP


class DTDSimCell(nn.Module):
    def __init__(self,  input_size=4, param_size=28, mlp_size=16, latent_size=5, num_hidden_layer=2):
        super(DTDSimCell, self).__init__()
        mlp_struct=[mlp_size]*num_hidden_layer
        mlp_struct.append(param_size)
        self.mlp2=MLP(latent_size,mlp_struct,\
                     activation_layer=nn.ReLU, inplace=None,dropout=0)
        self.B=nn.Parameter(torch.zeros(input_size-2,latent_size))
        self.A=nn.Parameter(torch.zeros(latent_size,latent_size))
    def forward(self,inputs,hidden):
        #basal_insulin_data,bolus_c_data,bolus_t_data,\
        #intake_c_data,intake_t_data,\
        #hr_data, hr_d_data, time_start_data, time_end_data
        insulin=inputs[:,0:1]; carb=inputs[:,1:2]
        other=inputs[:,2:]
        
        #dynamical states
        S=hidden[1]
        Gp,Gt=torch.split(S[:,:2],1,dim=-1) #glucose states
        Ip,Il=torch.split(S[:,2:4],1,dim=-1) #insulin states
        X,XL=torch.split(S[:,4:6],1,dim=-1) #intermediary states
        Qsto1,Qsto2,Qgut=torch.split(S[:,6:],1,dim=-1) #meal absorption states
        #H,SRSH,Hsc1,Hsc2=torch.split(S[:,16:20],1,dim=-1) #glucagon states
        
        #parameters
        Z=hidden[0]
        new_Z=torch.matmul(Z,self.A)+torch.matmul(other,self.B)
        K=torch.abs(self.mlp2(new_Z))
        
        
        k1,k2=torch.split(K[:,0:2],1,dim=-1) #glucose params
        m1,m2,m3,m4=torch.split(K[:,2:6],1,dim=-1) #insulin params 
        kgri,D,kmin,kmax,kabs,alpha,beta,b,c,D,BW,f=torch.split(K[:,6:18],1,dim=-1) #ra params
        kp1,kp2,kp3,ki=torch.split(K[:,18:22],1,dim=-1) #EGP params
        Uii,Vm0,Vmx,Km0,r1,p2u=torch.split(K[:,22:],1,dim=-1) # U params
        #ke1,ke2=torch.split(K[:,35:37],1,dim=-1) #E params
        #ka1,ka2,kd=torch.split(K[:,35:39],1,dim=-1) #sub insulin/glucose params
        #n,delta,rho,sig1,sig2,srbh,kh1,kh2,kh3,SRBH=torch.split(K[:,41:51],1,dim=-1) #glucagon params
        
        #static states computation
        #EGP system
        EGP=kp1-kp2*Gp-kp3*XL#+xi*XH
        #Ra system
        Qsto=Qsto1+Qsto2        
        kemptQ=kmin+(kmax-kmin/2)*(\
               torch.tanh(alpha*(Qsto-b*D))-\
               torch.tanh(beta*(Qsto-c*D))+2)
        Ra=f*kabs*Qgut/BW
        
        #Utilization system
        Uid=(Vm0+Vmx*X*r1)*Gt/(Km0+torch.abs(Gt))
        
        #renal excretion system
        #E=ke1*torch.nn.functional.relu(Gp-ke2)
        
        #subcutaneous insulin kinetics
        #Rai=ka1*Isc1+ka2*Isc2
                
        #dynamic states update
        #glucagon states
        #DSRSH=-rho*(SRSH-torch.nn.functional.relu(sig*(Gth-G)+SRBH))
        #DHsc1=-(kh1+kh2)*Hsc1
        #DHsc2=kh1*Hsc1-kh3*Hsc2
        #DH=-n*H+SRH+RaH
        #DXH=-kH*XH+kH*torch.nn.functional.relu(H-Hb)
        
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
        #DIr=-ki*(Ir-I)
        #DIsc1=-(kd+ka1)*Isc1+insulin
        #DIsc2=kd*Isc1-ka2*Isc2
        
        #glucose
        DGp=EGP+Ra-Uii-k1*Gp+k2*Gt
        DGt=-Uid+k1*Gp-k2*Gt
        #DGs=-Ts*Gs+Ts*G
        
        new_S=torch.concat([Gp+DGp, Gt+DGt,\
                        Ip+DIp, Il+DIl,\
                        X+DX, XL+DXL,\
                        Qsto1+DQsto1, Qsto2+DQsto2, Qgut+DQgut],axis=-1)

        return Gp+DGp, [new_Z,new_S]
    
class DTD_LSTM(nn.Module):
    def __init__(self,input_size=4,latent_size=8,state_size=9,\
                 output_size=1,mlp_size=16,num_hidden_layer=2):
        super(DTD_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                  hidden_size=latent_size,\
                  num_layers=2,\
                  batch_first=True)
        mlp_struct=[mlp_size]*num_hidden_layer
        mlp_struct.append(state_size-1)
        self.mlp1 = MLP(latent_size,mlp_struct,\
                     activation_layer=nn.ReLU, inplace=None,dropout=0)
        self.dtdcell=DTDSimCell(mlp_size=mlp_size,latent_size=latent_size,num_hidden_layer=num_hidden_layer)
    def forward(self,past,s,x):
        #past is N*L*5
        lstm_out, (h0,c0)=self.lstm(past)
        hidden=[c0[0],torch.concat([s[:,0],self.mlp1(h0[0])],axis=-1)]
        pred, hidden = self.dtdcell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.dtdcell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
        return pred
