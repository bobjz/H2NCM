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

GPU_ID=8
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
    
mod_name="Tran_OOD"
beta=1e4
for alpha in [0,1e-4,1e-3,1e-2,1e-1,1]:
    rmse=[]
    er=[]
    best_param_list=[]
    for repeat in range(3):
        for test_split in range(6):
            seed=repeat+2023
            d_model=[4,8]
            num_enc=[2,3]
            num_dec=[2,3]
            dim_ffd=[32,64]
            dropout=[0,0.1]
            hyper_params=[d_model,num_enc,num_dec,dim_ffd,dropout]
            list_params=np.array(list(itertools.product(*hyper_params)))
            scores=np.zeros(list_params.shape[0])
            for i in range(len(list_params)):
                 #tune hyperparams with cv
                params=list_params[i]
                for val_split in range(3):
                    torch.manual_seed(seed)
                    train,val,test,train_mean,train_std=cv_split2(perms,cases,ranks,repeat,test_split,val_split,batch_size=72,\
                                                                 train_intervention="insulin_carb", test_intervention="ins_carb_ratio")
                    model=Transformer(d_model=int(params[0]),num_encoder_layers=int(params[1]),\
                                     num_decoder_layers=int(params[2]), dim_feedforward=int(params[3]), dropout=params[4],device=device)
                    #total_params = sum(p.numel() for p in model.parameters())
                    #print(total_params)
                    train_h,val_h,test_h=train_trans(model,alpha,beta,train,val,test,epochs=100,lr=2*1e-3,device=device)
                    scores[i]+=np.min(val_h)/3

            best_param=list_params[np.argmin(scores)]
            print(f"best_param is {best_param}")
            best_param_list.append(best_param)
            torch.manual_seed(seed)
            train,val,test,train_mean,train_std=cv_split2(perms,cases,ranks,repeat,test_split,3,batch_size=72,\
                                                                 train_intervention="insulin_carb", test_intervention="ins_carb_ratio")
            model=Transformer(d_model=int(best_param[0]),num_encoder_layers=int(best_param[1]),\
                              num_decoder_layers=int(best_param[2]), dim_feedforward=int(best_param[3]),\
                              dropout=best_param[4],device=device)
            train_h,val_h,test_h=train_trans(model,alpha,beta,train,val,test,epochs=100,lr=2*1e-3,\
                                            device=device,path=f"{mod_name}_{alpha}_{repeat}_{test_split}.pth")
            print(f"repeat {repeat} test_split {test_split} pred{train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0])} causal{test_h[np.argmin(val_h)][1]}")
            rmse.append(train_std[0]*np.sqrt(test_h[np.argmin(val_h)][0]))
            er.append(test_h[np.argmin(val_h)][1])

    np.save(f"{mod_name}_a{alpha}_pred.npy",rmse)
    np.save(f"{mod_name}_a{alpha}_causal.npy",er)
    np.save(f"{mod_name}_a{alpha}_best_params.npy",best_param_list)
    print(f"{mod_name}_{alpha} RMSE {np.mean(rmse)}")
    print(f"{mod_name}_{alpha} Classification Error Rate {np.sort(er)}")
