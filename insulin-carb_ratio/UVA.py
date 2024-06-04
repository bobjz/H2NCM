import itertools
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.ops import MLP
from UVA_model import *
from utils import *
torch.set_default_dtype(torch.float64)
device=None
#comment this out if not using GPU
GPU_ID=9
device = torch.device('cuda:'+str(GPU_ID) if torch.cuda.is_available() else 'cpu')
print(device)


cases=np.load("/dfs/scratch1/bobjz/ICML_paper_data/new_icml_cases.npy")
ranks=np.load("/dfs/scratch1/bobjz/ICML_paper_data/new_icml_ranks.npy").astype("float64")

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
    for repeat in range(1): #change to 3 for alpha=1
        for test_split in range(6):
            seed=2023+repeat
            torch.manual_seed(2023)
            train,val,test,train_mean,train_std=cv_split2(perms,cases,ranks,repeat,test_split,3,batch_size=72,\
                                                                 train_intervention="insulin_carb", test_intervention="ins_carb_ratio")
            model=UVA_LSTM()
            train_h,val_h,test_h=train_model(model,alpha,beta,train,val,test,epochs=150,lr=1*1e-2,\
                                             device=device,path=f"UVA_{alpha}_{repeat}_{test_split}.pth")
            print(f"repeat {repeat} test_split {test_split} pred{train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0])} causal{test_h[np.argmin(val_h)][1]}")
            rmse.append(train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0]))
            er.append(test_h[np.argmin(val_h)][1])

    np.save(f"UVA_a{alpha}_pred.npy",rmse)
    np.save(f"UVA_a{alpha}_causal.npy",er)
    np.save(f"UVA_a{alpha}_best_params.npy",best_param_list)
    print(f"UVA_{alpha} RMSE {np.mean(rmse)}")
    print(f"UVA_{alpha} Classification Error Rate {np.sort(er)}")
