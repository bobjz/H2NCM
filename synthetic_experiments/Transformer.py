import numpy as np
import itertools 
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.ops import MLP
from Transformer_model import *
from utils import *
torch.set_default_dtype(torch.float64)

GPU_ID=2
device = torch.device('cuda:'+str(GPU_ID) if torch.cuda.is_available() else 'cpu')
print(device)


cases=np.float64(np.load("synthetic_cases.npy"))
ranks=np.float64(np.load("synthetic_ranks.npy"))

print(cases.shape)
print(ranks.shape)

repeats=3
rng=np.random.default_rng(seed=2024)
perms=np.zeros((repeats,cases.shape[0]),dtype='int32')
for i in range(repeats):
    perms[i]=rng.permutation(cases.shape[0])
    
beta=1e4

for alpha in [0,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1]:
    rmse=[]
    er=[]
    best_param_list=[]
    for repeat in range(2):
        for test_split in range(5):
            d_model=[4,8]
            num_enc=[2]
            num_dec=[2]
            dim_ffd=[32,64]
            dropout=[0,0.2]
            hyper_params=[d_model,num_enc,num_dec,dim_ffd,dropout]
            list_params=np.array(list(itertools.product(*hyper_params)))
            scores=np.zeros(list_params.shape[0])
            for i in range(len(list_params)):
                 #tune hyperparams with cv
                params=list_params[i]
                for val_split in range(3):
                    torch.manual_seed(2023)
                    train,val,test,train_mean,train_std=cv_split(perms,cases,ranks,repeat,test_split,val_split,batch_size=200)
                    model=Transformer(d_model=int(params[0]),num_encoder_layers=int(params[1]),\
                                     num_decoder_layers=int(params[2]), dim_feedforward=int(params[3]), dropout=params[4],device=device)
                    #total_params = sum(p.numel() for p in model.parameters())
                    #print(total_params)
                    train_h,val_h,test_h=train_trans(model,alpha,beta,train,val,test,epochs=50,lr=10*1e-3,device=device)
                    scores[i]+=np.min(val_h)/3

            best_param=list_params[np.argmin(scores)]
            print(f"best_param is {best_param}")
            best_param_list.append(best_param)
            torch.manual_seed(2023)
            train,val,test,train_mean,train_std=cv_split(perms,cases,ranks,repeat,test_split,3,batch_size=200)
            model=Transformer(d_model=int(best_param[0]),num_encoder_layers=int(best_param[1]),\
                              num_decoder_layers=int(best_param[2]), dim_feedforward=int(best_param[3]),\
                              dropout=best_param[4],device=device)
            train_h,val_h,test_h=train_trans(model,alpha,beta,train,val,test,epochs=50,lr=10*1e-3,\
                                            device=device,path=f"Tran_{alpha}_{repeat}_{test_split}.pth")
            print(f"repeat {repeat} test_split {test_split} pred{train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0])} causal{test_h[np.argmin(val_h)][1]}")
            rmse.append(train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0]))
            er.append(test_h[np.argmin(val_h)][1])

    np.save(f"Tran_a{alpha}_pred.npy",rmse)
    np.save(f"Tran_a{alpha}_causal.npy",er)
    np.save(f"Tran_a{alpha}_best_params.npy",best_param_list)
    print(f"Tran_{alpha} RMSE {np.mean(rmse)}")
    print(f"Tran_{alpha} Classification Error Rate {np.sort(er)}")
