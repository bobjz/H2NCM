import itertools
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.ops import MLP
from LP_model import *
from utils import *
torch.set_default_dtype(torch.float64)
device=None
#comment this out if not using GPU
GPU_ID=8
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
#tune hyperparam
for alpha in [0,1e-4,1e-3,1e-2,1e-1,1]:
    rmse=[]
    er=[]
    best_param_list=[]
    for repeat in range(3):
        for test_split in range(5):
            num_hidden_layers=[2,3]
            mlp_size=[16,32]
            latent_size=[2,4]
            hyper_params=[num_hidden_layers,mlp_size,latent_size]
            list_params=np.array(list(itertools.product(*hyper_params)))
            scores=np.zeros(list_params.shape[0])
            for i in range(len(list_params)):
                #tune hyperparams with cv
                params=list_params[i]
                for val_split in range(3):
                    torch.manual_seed(2023)
                    train,val,test,train_mean,train_std=cv_split(perms,cases,ranks,repeat,test_split,val_split,batch_size=64)
                    model=DTD_LSTM(latent_size=int(params[2]),mlp_size=int(params[1]),num_hidden_layer=int(params[0]))
                    #total_params = sum(p.numel() for p in model.parameters())
                    #print(total_params)
                    train_h,val_h,test_h=train_model(model,alpha,beta,train,val,test,epochs=50,lr=10*1e-3,device=device)
                    scores[i]+=np.min(val_h)/3

            best_param=list_params[np.argmin(scores)]
            print(f"best_param is {best_param}")
            best_param_list.append(best_param)
            torch.manual_seed(2023)
            train,val,test,train_mean,train_std=cv_split(perms,cases,ranks,repeat,test_split,3,batch_size=64)
            model=DTD_LSTM(latent_size=int(best_param[2]),mlp_size=int(best_param[1]),num_hidden_layer=int(best_param[0]))
            train_h,val_h,test_h=train_model(model,alpha,beta,train,val,test,epochs=50,lr=10*1e-3,\
                                             device=device,path=f"LP_{alpha}_{repeat}_{test_split}.pth")
            print(f"repeat {repeat} test_split {test_split} pred{train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0])} causal{test_h[np.argmin(val_h)][1]}")
            rmse.append(train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0]))
            er.append(test_h[np.argmin(val_h)][1])

    np.save(f"LP_a{alpha}_pred.npy",rmse)
    np.save(f"LP_a{alpha}_causal.npy",er)
    np.save(f"LP_a{alpha}_best_params.npy",best_param_list)
    print(f"LP_{alpha} RMSE {np.mean(rmse)}")
    print(f"LP_{alpha} Classification Error Rate {np.sort(er)}")
