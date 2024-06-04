import numpy as np
import itertools 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.ops import MLP
from TCN_model import *
from utils import *
torch.set_default_dtype(torch.float64)
device=None
#comment this out if not using GPU
GPU_ID=6
device = torch.device('cuda:'+str(GPU_ID) if torch.cuda.is_available() else 'cpu')
print(device)


cases=np.load("/dfs/scratch1/bobjz/ICML_paper_data/Final_T1DEXI_CASES.npy")
ranks=np.load("/dfs/scratch1/bobjz/ICML_paper_data/Final_T1DEXI_RANKS.npy")

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
        for test_split in range(6):
            num_layers=[2,3]
            conv_size=[16,24,32]
            kernel_size=[2,3,4]
            dropout=[0,0.1,0.2]
            hyper_params=[num_layers,conv_size,kernel_size,dropout]
            list_params=np.array(list(itertools.product(*hyper_params)))
            scores=np.zeros(list_params.shape[0])
            for i in range(len(list_params)):
                #tune hyperparams with cv
                params=list_params[i]
                for val_split in range(3):
                    torch.manual_seed(2023)
                    train,val,test,train_mean,train_std=cv_split2(perms,cases,ranks,repeat,test_split,val_split,batch_size=64,\
                                                                 train_intervention="insulin_carb", test_intervention="inscarb_ratio")
                    model=TCN_wrapper(5,6, num_channels=[int(params[1])]*int(params[0]),\
                                      kernel_size=int(params[2]), dropout=params[3])
                    #total_params = sum(p.numel() for p in model.parameters())
                    #print(total_params)
                    train_h,val_h,test_h=train_model(model,alpha,beta,train,val,test,epochs=100,lr=2*1e-3,device=device)
                    scores[i]+=np.min(val_h)/3

            best_param=list_params[np.argmin(scores)]
            print(f"best_param is {best_param}")
            best_param_list.append(best_param)
            torch.manual_seed(2023)
            train,val,test,train_mean,train_std=cv_split2(perms,cases,ranks,repeat,test_split,3,batch_size=64,\
                                                                 train_intervention="insulin_carb", test_intervention="inscarb_ratio")
            model=TCN_wrapper(5,6, num_channels=[int(best_param[1])]*int(best_param[0]),\
                                      kernel_size=int(best_param[2]), dropout=best_param[3])
            train_h,val_h,test_h=train_model(model,alpha,beta,train,val,test,epochs=100,lr=2*1e-3,\
                                             device=device,path=f"TCN_{alpha}_{repeat}_{test_split}.pth")
            print(f"repeat {repeat} test_split {test_split} pred{train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0])} causal{test_h[np.argmin(val_h)][1]}")
            rmse.append(train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0]))
            er.append(test_h[np.argmin(val_h)][1])

    np.save(f"TCN_a{alpha}_pred.npy",rmse)
    np.save(f"TCN_a{alpha}_causal.npy",er)
    np.save(f"TCN_a{alpha}_best_params.npy",best_param_list)
    print(f"TCN_{alpha} RMSE {np.mean(rmse)}")
    print(f"TCN_{alpha} Classification Error Rate {np.sort(er)}")