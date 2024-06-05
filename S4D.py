import numpy as np
import itertools 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.ops import MLP
from S4D_model import *
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

mod_name="S4D"
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
            seed=repeat+2023
            d_model=[4,6,8]
            d_state=[32,64]
            dropout=[0,0.1,0.2]
            hyper_params=[d_model,d_state,dropout]
            list_params=np.array(list(itertools.product(*hyper_params)))
            scores=np.zeros(list_params.shape[0])
            for i in range(len(list_params)):
                #tune hyperparams with cv
                params=list_params[i]
                for val_split in range(3):
                    torch.manual_seed(seed)
                    train,val,test,train_mean,train_std=cv_split(perms,cases,ranks,repeat,test_split,val_split,batch_size=72)
                    model=S4D_wrapper(d_model=int(params[0]), d_state=int(params[1]), dropout=params[2])
                    #total_params = sum(p.numel() for p in model.parameters())
                    #print(total_params)
                    train_h,val_h,test_h=train_model(model,alpha,beta,train,val,test,epochs=100,lr=2*1e-3,device=device)
                    scores[i]+=np.min(val_h)/3

            best_param=list_params[np.argmin(scores)]
            print(f"best_param is {best_param}")
            best_param_list.append(best_param)
            torch.manual_seed(seed)
            train,val,test,train_mean,train_std=cv_split(perms,cases,ranks,repeat,test_split,3,batch_size=72)
            model=S4D_wrapper(d_model=int(best_param[0]), d_state=int(best_param[1]), dropout=best_param[2])
            train_h,val_h,test_h=train_model(model,alpha,beta,train,val,test,epochs=100,lr=2*1e-3,\
                                             device=device,path=f"{mod_name}_{alpha}_{repeat}_{test_split}.pth")
            print(f"repeat {repeat} test_split {test_split} pred{train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0])} causal{test_h[np.argmin(val_h)][1]}")
            rmse.append(train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0]))
            er.append(test_h[np.argmin(val_h)][1])

    np.save(f"{mod_name}_a{alpha}_pred.npy",rmse)
    np.save(f"{mod_name}_a{alpha}_causal.npy",er)
    np.save(f"{mod_name}_a{alpha}_best_params.npy",best_param_list)
    print(f"{mod_name}_{alpha} RMSE {np.mean(rmse)}")
    print(f"S4D_{alpha} Classification Error Rate {np.sort(er)}")
