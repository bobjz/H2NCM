import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FormDataset(Dataset):
    """
    Standerd Pytoch Dataset Constructor
    """
    def __init__(self, cases, ranks, window=6):        
        self.cases = cases
        self.ranks = ranks
        self.window = window

    def __len__(self):
        return len(self.ranks)

    def __getitem__(self, idx):
        window=self.window
        if torch.is_tensor(idx):
            idx = idx.tolist()
        history=self.cases[idx,:,:-window-1]
        future_cov=self.cases[idx,:,-window-1:-1,1:]
        starting_cgm=self.cases[idx,:,-window-1:-window,0:1]
        output_cgm=self.cases[idx,:,-window:,0:1]
        rankings=self.ranks[idx]
        
        sample = (history,starting_cgm,future_cov,\
                  output_cgm,rankings)
        return sample
    

def cv_split(perms,cases,ranks,rep,outer_fold,inner_fold,ts=12,vs=21,batch_size=16):
#split and form data loarder
    perm=perms[rep]
    cases_s=cases[perm]
    ranks_s=ranks[perm]
    
    test_cases=cases_s[outer_fold*ts:(outer_fold+1)*ts]
    test_ranks=ranks_s[outer_fold*ts:(outer_fold+1)*ts]
    cv_cases=np.concatenate([cases_s[0:outer_fold*ts],cases_s[(outer_fold+1)*ts:]],axis=0)
    cv_ranks=np.concatenate([ranks_s[0:outer_fold*ts],ranks_s[(outer_fold+1)*ts:]],axis=0)
    
    val_cases=cv_cases[inner_fold*vs:(inner_fold+1)*vs]
    val_ranks=cv_ranks[inner_fold*vs:(inner_fold+1)*vs]
    train_cases=np.concatenate([cv_cases[0:inner_fold*vs],cv_cases[(inner_fold+1)*vs:]],axis=0)
    train_ranks=np.concatenate([cv_ranks[0:inner_fold*vs],cv_ranks[(inner_fold+1)*vs:]],axis=0)
    
    train_mean=np.mean(train_cases[:,0],axis=(0,1))
    train_std=np.std(train_cases[:,0],axis=(0,1))
    train_cases=np.divide(train_cases-train_mean,train_std)
    val_cases=np.divide(val_cases-train_mean,train_std)
    test_cases=np.divide(test_cases-train_mean,train_std)
    
    train=DataLoader(FormDataset(train_cases,train_ranks),\
                                   batch_size=batch_size)
    val=DataLoader(FormDataset(val_cases,val_ranks),\
                                   batch_size=len(val_ranks))
    test=DataLoader(FormDataset(test_cases,test_ranks),\
                                   batch_size=len(test_ranks))

    return train,val,test,train_mean,train_std


def cv_split2(perms,cases,ranks,rep,outer_fold,inner_fold,ts=12,vs=21,batch_size=16,\
             train_intervention=None, test_intervention=None, corruption=0):
#split and form data loarder
#This function is for additional experiments that involve modifications of the interventions sets
    rng=np.random.default_rng(2024)

    perm=perms[rep]
    cases_s=cases[perm]
    ranks_s=ranks[perm]
    
    test_cases=cases_s[outer_fold*ts:(outer_fold+1)*ts]
    test_ranks=ranks_s[outer_fold*ts:(outer_fold+1)*ts]
    cv_cases=np.concatenate([cases_s[0:outer_fold*ts],cases_s[(outer_fold+1)*ts:]],axis=0)
    cv_ranks=np.concatenate([ranks_s[0:outer_fold*ts],ranks_s[(outer_fold+1)*ts:]],axis=0)
    
    #modeify interventions
    if train_intervention=="insulin_carb":
        for i in range(len(cv_cases)):
            for j in range(1,4):
                cv_cases[i,j]=cv_cases[i,0]
            roll=rng.random()
            if roll<0.5:
                cv_cases[i,2,-7:,1]+=2.5
                cv_cases[i,3,-7:,1]+=5.0
                cv_ranks[i]=[1,0,0]
            else:
                cv_cases[i,2,-7,2]+=50.0
                cv_cases[i,3,-7,2]+=100.0
                cv_ranks[i]=[0,0,1]
    if test_intervention=="inscarb_ratio":
        for i in range(len(test_cases)):
            for j in range(1,4):
                test_cases[i,j]=test_cases[i,0]
            test_cases[i,1,-7,1]+=2.25
            test_cases[i,2,-7,1]+=3.00
            test_cases[i,3,-7,1]+=4.50
            test_cases[i,1:,-7,2]+=45
            test_ranks[i]=[1,0,0]
    #apply corruption
    #if corruption>0:
    #    for i in range(len(cv_ranks)):
    #        rng=np.random.default_rng(2024)
    #        roll=rng.random()
    #        if roll<corruption:
    #            cv_ranks[i]=np.roll(cv_ranks[i],1)
    val_cases=cv_cases[inner_fold*vs:(inner_fold+1)*vs]
    val_ranks=cv_ranks[inner_fold*vs:(inner_fold+1)*vs]
    train_cases=np.concatenate([cv_cases[0:inner_fold*vs],cv_cases[(inner_fold+1)*vs:]],axis=0)
    train_ranks=np.concatenate([cv_ranks[0:inner_fold*vs],cv_ranks[(inner_fold+1)*vs:]],axis=0)
    
    train_mean=np.mean(train_cases[:,0],axis=(0,1))
    train_std=np.std(train_cases[:,0],axis=(0,1))
    train_cases=np.divide(train_cases-train_mean,train_std)
    val_cases=np.divide(val_cases-train_mean,train_std)
    test_cases=np.divide(test_cases-train_mean,train_std)
    
    train=DataLoader(FormDataset(train_cases,train_ranks),\
                                   batch_size=batch_size)
    val=DataLoader(FormDataset(val_cases,val_ranks),\
                                   batch_size=len(val_ranks))
    test=DataLoader(FormDataset(test_cases,test_ranks),\
                                   batch_size=len(test_ranks))

    return train,val,test,train_mean,train_std



def train_model(model,alpha,beta,train,val,test,epochs,lr,device=None, verbose=False, path=None):
    #train/validate model with train and val, save to path
    if (device):
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn1=nn.MSELoss()
    loss_fn2=nn.CrossEntropyLoss()
    train_losses=[]
    val_losses=[]
    test_losses=[]
    best_val=1e7
    for epoch in range(epochs):
        model.train()
        train_loss1=0
        train_loss2=0
        for batch, (past,y0,x,y,rank) in enumerate(train):
            if device:
                past=past.to(device)
                y0=y0.to(device)
                x=x.to(device)
                y=y.to(device)
                rank=rank.to(device)
            preds=[]
            for i in range(4):
                preds.append(model(past[:,i],y0[:,i],x[:,i]))
            rank_bg=torch.concat([torch.mean(pred[:,:,0],dim=-1,keepdim=True) for pred in preds[1:]],dim=-1)
            pred_rank=nn.functional.softmax(rank_bg*beta,dim=-1)
            loss1 = loss_fn1(preds[0], y[:,0])
            loss2 = loss_fn2(torch.log(pred_rank+1e-7),rank)
            loss = (1-alpha)*loss1+alpha*loss2
            if verbose:
                print(f"training loss 1 {loss1.item()} loss 2 {loss2.item()}")
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            train_loss1+=loss1.item()*len(rank)
            train_loss2+=loss2.item()*len(rank)
        train_size=len(train.dataset)
        train_losses.append([train_loss1/train_size,train_loss2/train_size])

        model.eval()
        with torch.no_grad():

            for batch, (past,y0,x,y,rank) in enumerate(val):
                if device:
                    past=past.to(device)
                    y0=y0.to(device)
                    x=x.to(device)
                    y=y.to(device)
                    rank=rank.to(device)
                preds=[]
                for i in range(4):
                    preds.append(model(past[:,i],y0[:,i],x[:,i]))
                rank_bg=torch.concat([torch.mean(pred[:,:,0],dim=-1,keepdim=True) for pred in preds[1:]],dim=-1)
                pred_rank=nn.functional.softmax(rank_bg*beta,dim=-1)
                loss1 = loss_fn1(preds[0], y[:,0])
                loss2 = loss_fn2(torch.log(pred_rank+1e-7),rank)
                loss_val = (1-alpha)*loss1+alpha*loss2  
                valid_loss = loss_val.item()
                val_losses.append(valid_loss)
                if valid_loss<best_val and path:
                    best_val=valid_loss
                    torch.save(model.state_dict(),path)
                if verbose:
                    print(f"validation loss at epoch {epoch} pred {loss1.item()} causal {loss2.item()}")

            for batch, (past,y0,x,y,rank) in enumerate(test):
                if device:
                    past=past.to(device)
                    y0=y0.to(device)
                    x=x.to(device)
                    y=y.to(device)
                    rank=rank.to(device)
                preds=[]
                for i in range(4):
                    preds.append(model(past[:,i],y0[:,i],x[:,i]))
                rank_bg=torch.concat([torch.mean(pred[:,:,0],dim=-1,keepdim=True) for pred in preds[1:]],dim=-1)
                pred_rank=nn.functional.softmax(rank_bg*beta,dim=-1)
                pred_rank2=nn.functional.softmax(rank_bg*1e7,dim=-1)
                loss1 = loss_fn1(preds[0], y[:,0])
                test_losses.append([loss1.item(),round(torch.sum(torch.abs(pred_rank2-rank)).item()/2/len(rank),3)])
                if verbose:
                    print(f"test loss at epoch {epoch} pred {loss1.item()} causal {round(torch.sum(torch.abs(pred_rank2-rank)).item()/2/len(rank),3)}")
                
    return train_losses, val_losses, test_losses

def train_trans(model,alpha,beta,train,val,test,epochs,lr,device=None, verbose=False, path=None):
    #train/validate model with train and val, save to path
    if (device):
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn1=nn.MSELoss()
    loss_fn2=nn.CrossEntropyLoss()
    train_losses=[]
    val_losses=[]
    test_losses=[]
    best_val=1e7
    for epoch in range(epochs):
        model.train()
        train_loss1=0
        train_loss2=0
        for batch, (past,y0,x,y,rank) in enumerate(train):
            if device:
                past=past.to(device)
                y0=y0.to(device)
                x=x.to(device)
                y=y.to(device)
                rank=rank.to(device)
            preds=[]
            causal_mask=nn.Transformer.generate_square_subsequent_mask(x.shape[-2],device=device)
            for i in range(4):
                src_x=nn.functional.pad(x[:,i],(1,0),'constant',0)
                src_x[:,0:1,0:1]=y0[:,i]
                src=torch.concat([past[:,i],src_x],dim=1)
                tgt=torch.concat([y0[:,i],y[:,i,:-1]],dim=1)
                preds.append(model(src,tgt,causal_mask))
            rank_bg=torch.concat([torch.mean(pred[:,:,0],dim=-1,keepdim=True) for pred in preds[1:]],dim=-1)
            pred_rank=nn.functional.softmax(rank_bg*beta,dim=-1)
            loss1 = loss_fn1(preds[0], y[:,0])
            loss2 = loss_fn2(torch.log(pred_rank+1e-7),rank)
            loss = (1-alpha)*loss1+alpha*loss2   
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            train_loss1+=loss1.item()*len(rank)
            train_loss2+=loss2.item()*len(rank)
        train_size=len(train.dataset)
        train_losses.append([train_loss1/train_size,train_loss2/train_size])

        model.eval()
        with torch.no_grad():

            for batch, (past,y0,x,y,rank) in enumerate(val):
                if device:
                    past=past.to(device)
                    y0=y0.to(device)
                    x=x.to(device)
                    y=y.to(device)
                    rank=rank.to(device)
                preds=[]
                for i in range(4):
                    src_x=nn.functional.pad(x[:,i],(1,0),'constant',0)
                    src_x[:,0:1,0:1]=y0[:,i]
                    src=torch.concat([past[:,i],src_x],dim=1)
                    tgt=torch.concat([y0[:,i],y[:,i,:-1]*0],dim=1)
                    prediction=model(src,tgt)
                    for j in range(1,x.shape[-2]):
                        tgt[:,j]=prediction[:,j-1]
                        prediction=model(src,tgt)
                    preds.append(prediction)
                rank_bg=torch.concat([torch.mean(pred[:,:,0],dim=-1,keepdim=True) for pred in preds[1:]],dim=-1)
                pred_rank=nn.functional.softmax(rank_bg*beta,dim=-1)
                loss1 = loss_fn1(preds[0], y[:,0])
                loss2 = loss_fn2(torch.log(pred_rank+1e-7),rank)
                loss_val = (1-alpha)*loss1+alpha*loss2  
                valid_loss = loss_val.item()
                val_losses.append(valid_loss)
                if valid_loss<best_val and path:
                    best_val=valid_loss
                    torch.save(model.state_dict(),path)
                if verbose:
                    print(f"validation loss at epoch {epoch} pred {loss1.item()} causal {loss2.item()}")

            for batch, (past,y0,x,y,rank) in enumerate(test):
                if device:
                    past=past.to(device)
                    y0=y0.to(device)
                    x=x.to(device)
                    y=y.to(device)
                    rank=rank.to(device)
                preds=[]
                for i in range(4):
                    src_x=nn.functional.pad(x[:,i],(1,0),'constant',0)
                    src_x[:,0:1,0:1]=y0[:,i]
                    src=torch.concat([past[:,i],src_x],dim=1)
                    tgt=torch.concat([y0[:,i],y[:,i,:-1]*0],dim=1)
                    prediction=model(src,tgt)
                    for j in range(1,x.shape[-2]):
                        tgt[:,j]=prediction[:,j-1]
                        prediction=model(src,tgt)
                    preds.append(prediction)
                rank_bg=torch.concat([torch.mean(pred[:,:,0],dim=-1,keepdim=True) for pred in preds[1:]],dim=-1)
                pred_rank=nn.functional.softmax(rank_bg*beta,dim=-1)
                pred_rank2=nn.functional.softmax(rank_bg*1e7,dim=-1)
                loss1 = loss_fn1(preds[0], y[:,0])
                test_losses.append([loss1.item(),round(torch.sum(torch.abs(pred_rank2-rank)).item()/2/len(rank),3)])
                
    return train_losses, val_losses, test_losses