import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchvision import transforms, utils
from torchvision.ops import MLP


class DTDSimCell(nn.Module):
    def __init__(self,  input_size=4, param_size=53, mlp_size=16, latent_size=8, num_hidden_layer=2):
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
        Gp,Gt,Gs,DGp=torch.split(S[:,:4],1,dim=-1) #glucose states
        Ip,Il,DIp,Isc1,Isc2=torch.split(S[:,4:9],1,dim=-1) #insulin states
        X,XL,XH,Ir=torch.split(S[:,9:13],1,dim=-1) #intermediary states
        Qsto1,Qsto2,Qgut=torch.split(S[:,13:16],1,dim=-1) #meal absorption states
        H,SRSH,Hsc1,Hsc2=torch.split(S[:,16:],1,dim=-1) #glucagon states
        
        
        #parameters
        Z=hidden[0]
        new_Z=torch.matmul(Z,self.A)+torch.matmul(other,self.B)
        K=torch.abs(self.mlp2(new_Z))
        
        k1,k2,VG=torch.split(K[:,0:3],1,dim=-1) #glucose params
        m1,m2,m3,m4,Vl=torch.split(K[:,3:8],1,dim=-1) #insulin params 
        kgri,D,kmin,kmax,kabs,alpha,beta,b,c,D,BW,f=torch.split(K[:,8:20],1,dim=-1) #ra params
        kp1,kp2,kp3,xi,ki,kH,Hb=torch.split(K[:,20:27],1,dim=-1) #EGP params
        Uii,Vm0,Vmx,Km0,r1,r2,p2u,Ib=torch.split(K[:,27:35],1,dim=-1) # U params
        ke1,ke2=torch.split(K[:,35:37],1,dim=-1) #E params
        ka1,ka2,kd,Ts=torch.split(K[:,37:41],1,dim=-1) #sub insulin/glucose params
        n,delta,rho,sig1,sig2,srbh,kh1,kh2,kh3,SRBH,Gb,Gth=torch.split(K[:,41:],1,dim=-1) #glucagon params
        
        #constants and convinence states
        G=Gp/VG;I=Ip/Vl
        #Gb=0.5
        #Gth=0.25
        ratio=nn.functional.relu(G/Gb)
        
        
        #static states computation
        
        #Glucagon system
        SRDH=delta*nn.functional.relu(-DGp,0)
        pred=torch.sigmoid(10000*(ratio-1))
        sig=(1-pred)*sig1/(nn.functional.relu(I)+0.1)+pred*sig2
        SRH=SRSH+SRDH
        RaH=kh3*Hsc2
        
        #EGP system
        EGP=kp1-kp2*Gp-kp3*XL+xi*XH

        #Ra system
        Qsto=Qsto1+Qsto2        
        kemptQ=kmin+(kmax-kmin/2)*(\
               torch.tanh(alpha*(Qsto-b*D))-\
               torch.tanh(beta*(Qsto-c*D))+2)
        Ra=f*kabs*Qgut/BW
        
        #Utilization system
        Uid=(Vm0+Vmx*X*r1)*Gt/(Km0+torch.abs(Gt))
        #renal excretion system
        E=ke1*nn.functional.relu(Gp-ke2)
        
        #subcutaneous insulin kinetics
        Rai=ka1*Isc1+ka2*Isc2
        
        
        #dynamic states update
        
        #glucagon states
        DSRSH=-rho*(SRSH-nn.functional.relu(sig*(Gth-G)+SRBH))
        DHsc1=-(kh1+kh2)*Hsc1
        DHsc2=kh1*Hsc1-kh3*Hsc2
        DH=-n*H+SRH+RaH
        DXH=-kH*XH+kH*nn.functional.relu(H-Hb)
        
        #Meal
        DQsto1=-kgri*Qsto1+D*carb
        DQsto2=-kemptQ*Qsto2+kgri*Qsto1
        DQgut=-kabs*Qgut+kemptQ*Qsto2
      
        #Utilization
        DX=-p2u*X+p2u*(I-Ib)
        
        #insulin 
        DIp=-(m2+m4)*Ip+m1*Il+Rai
        DIl=-(m1+m3)*Il+m2*Ip
        DIr=-ki*(Ir-I)
        DXL=-ki*(XL-Ir)
        DIsc1=-(kd+ka1)*Isc1+(insulin)
        DIsc2=kd*Isc1-ka2*Isc2
        
        #glucose
        DGp=EGP+Ra-Uii-E-k1*Gp+k2*Gt
        DGt=-Uid+k1*Gp-k2*Gt
        DGs=-Ts*Gs+Ts*G
        
        new_S=torch.concat([Gp+DGp, Gt+DGt, Gs+DGs, DGp,\
                        Ip+DIp, Il+DIl, DIp, Isc1+DIsc1, Isc2+DIsc2,\
                        X+DX, XL+DXL, XH+DXH, Ir+DIr,\
                        Qsto1+DQsto1, Qsto2+DQsto2, Qgut+DQgut,\
                        H+DH, SRSH+DSRSH, Hsc1+DHsc1, Hsc2+DHsc2],axis=-1)
        
        return Gs+DGs, [new_Z,new_S]
    
class DTD_LSTM(nn.Module):
    def __init__(self,input_size=4,latent_size=8,state_size=20,\
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
        hidden=[c0[1],torch.concat([s[:,0],self.mlp1(h0[1])],axis=-1)]
        pred, hidden = self.dtdcell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.dtdcell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
        return pred
