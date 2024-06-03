import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
pl=os.listdir("Patients")

np.set_printoptions(suppress=True)
def conv_time(time,offset=True):
    import time as dt
    if offset:
        time+=1.9e+09
    tm=dt.gmtime(time)
    return tm.tm_hour*1e+4+tm.tm_min*1e+2+tm.tm_sec


def conv_date(time,offset=True):
    import time as dt
    if offset:
        time+=1.9e+09
    tm=dt.gmtime(time)
    return tm.tm_mon*100+tm.tm_mday
def extract(series,time,method=1):
    ind=np.searchsorted(series[:,1],time)-method
    return series[ind,0]
def extract_ind(series,time):
    return np.searchsorted(series,time)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def process_bolus(bolus_time,bolus_val,rate):
    n=len(bolus_val)
    new_bolus_time=[]
    new_bolus_val=[]
    for i in range(n):
        if i==0:
            new_bolus_time.append(bolus_time[i])
            new_bolus_val.append(bolus_val[i])
            continue
        if bolus_time[i]<new_bolus_time[-1]+new_bolus_val[-1]/rate*60:
            new_bolus_val[-1]+=bolus_val[i]
        else:
            new_bolus_val.append(bolus_val[i])
            new_bolus_time.append(bolus_time[i])
    return (np.array(new_bolus_time), np.array(new_bolus_val))
def step_bolus(bolus_time,bolus_val,rate):
    if len(bolus_time)==0:
        return (np.array([-240*60.0,30*60.0]),np.array([0,0]))
    if bolus_time[0]!=-240*60.0:
        step_bolus_time=[-240*60.0]
        step_bolus_val=[0.0]
    for i in range(len(bolus_val)):
        if bolus_val[i]>0:
            step_bolus_time.append(bolus_time[i])
            step_bolus_val.append(rate)
            step_bolus_time.append(bolus_time[i]+bolus_val[i]/rate*60)
            step_bolus_val.append(0)
    if bolus_time[-1]>=30*60:
        step_bolus_time[-1]=30*60.0
        step_bolus_val[-1]=rate
    else:
        step_bolus_time.append(30*60.0)
        step_bolus_val.append(0)
    return (np.array(step_bolus_time),np.array(step_bolus_val))
def add_step(step_time1,step_val1,step_time2,step_val2):
    
    n2=len(step_time2)
    for i in range(n2-1):
        if step_val2[i]==0:
            continue
        t1=step_time2[i]
        t2=step_time2[i+1]
        
        if t2 not in step_time1:
            ind=np.searchsorted(step_time1,t2)
            step_time1=np.insert(step_time1,ind,t2)
            step_val1=np.insert(step_val1,ind,step_val1[ind-1])
        n1=len(step_time1)
        indices=np.array([i for i in range(n1)])
        ind=indices[(step_time1>=t1)&(step_time1<t2)]
        for j in ind:
            step_val1[j]=step_val1[j]+step_val2[i]
        if t1 not in step_time1:
            if len(ind)>0:
                step_time1=np.insert(step_time1,ind[0],t1)
                step_val1=np.insert(step_val1,ind[0],step_val1[ind[0]-1]+step_val2[i])
            else:
                ind=np.searchsorted(step_time1,t1)
                step_time1=np.insert(step_time1,ind,t1)
                step_val1=np.insert(step_val1,ind,step_val1[ind-1]+step_val2[i])

    step_val1[-1]=step_val1[-1]+step_val2[-1]
    return (step_time1, step_val1)

train_cases=[]
train_cov=[]
train_ranks=[]
rng=np.random.default_rng(2024)
cgm_count=[]
found=False
plot=False
for n in range(len(pl)):
    pat=pl[n]    
    if (pat[0]!='P'):
        continue
    #if (pat!='Patient_1121') and not found:
    #    continue
    #if (pat=='Patient_446'):
    #    found=True
    #print(pat)
    folder=f"Patients/{pat}/"
    PR=pd.read_csv(f"{folder}PR.csv")
    DM=pd.read_csv(f"{folder}DM.csv")
    age=DM.iloc[0,6]
    VS=pd.read_csv(f"{folder}VS.csv")

    weight=VS.iloc[-1,3]
    FADX=pd.read_csv(f"{folder}FADX.csv")
    print(pat)
    pat_count=0
    ##weight add later
    for ind,exe in PR.iterrows():
        if pat_count>1:
            break
        dice1=rng.random()
        dice2=rng.random()
        if exe.PLNEXDUR<30:
            continue
        ##start time
        s_time=exe.PRSTDTC+exe.PLNEXDUR*60 
        
        ##extract cgm
        LB=pd.read_csv(f"{folder}LB.csv")
        LB=LB[(s_time-240*60<LB.LBDTC) & (LB.LBDTC<s_time+30*60)]
        #LB=LB[(LB.LBDTC<s_time+30*60)]
        cgm=LB.LBSTRESN.to_numpy()
        time_stamp=LB.LBDTC.to_numpy()
        #LB2=LB[LB.LBDTC>s_time-240*60]
        if (time_stamp.shape[0]!=54) or (cgm.shape[0]!=54):
            continue
        #print(exe)
        ##carb
        if os.path.isfile(f"{folder}FAMLPM.csv"):
            FAML=pd.read_csv(f"{folder}FAMLPM.csv")
        else:
            continue
        ml=FAML[FAML.FATEST=="b'Dietary Total Carbohydrate'"]
        ml=ml[ml.FACAT=="b'CONSUMED'"]
        ml=ml[(ml.FADTC<s_time+30*60) & (ml.FADTC>s_time-240*60)]
        ml.sort_values(by=['FADTC'],inplace=True)
        res_carb=pd.read_csv(f"{folder}ML.csv")
        res_carb=res_carb[res_carb.MLCAT=="b'RESCUE CARBS'"]
        res_carb=res_carb[(res_carb.MLDTC<s_time+30*60) & (res_carb.MLDTC>s_time-240*60)]
        res_carb.sort_values(by=['MLDTC'],inplace=True)
        if (ml.empty) and (res_carb.empty):
            continue
        meal=np.zeros(54)
        for null,ml_dose in ml.iterrows():
            ml_time=ml_dose.FADTC
            ind=np.searchsorted(time_stamp,ml_time)
            if (ind==54):
                continue
            ml_inmg=float(ml_dose.FAORRES[2:-1])*1e3
            ml_endtime=ml_time+ml_inmg/45e3*60
            #print(time_stamp[ind-1])
            #print(time_stamp[ind])
            #print(dose_time)
            #print(dose_endtime)
            if ml_endtime<time_stamp[ind]:
                meal[ind-1]+=np.round(ml_inmg/5,3)
            else:
                meal[ind-1]+=np.round((time_stamp[ind]-ml_time)/60*45e3/5,3)
                while time_stamp[ind]<=ml_endtime:
                    meal[ind]+=np.round(min(ml_endtime-time_stamp[ind],300)/60*45e3/5,3)
                    ind+=1
                    if ind>53:
                        break
        for null,ml_dose in res_carb.iterrows():
            ml_time=ml_dose.MLDTC
            ind=np.searchsorted(time_stamp,ml_time)
            if (ind==54):
                continue
            ml_inmg=float(ml_dose.MLDOSE)*1e3
            ml_endtime=ml_time+ml_inmg/45e3*60
            #print(time_stamp[ind-1])
            #print(time_stamp[ind])
            #print(dose_time)
            #print(dose_endtime)
            if ml_endtime<time_stamp[ind]:
                meal[ind-1]+=np.round(ml_inmg/5,3)
            else:
                meal[ind-1]+=np.round((time_stamp[ind]-ml_time)/60*45e3/5,3)
                while time_stamp[ind]<=ml_endtime:
                    meal[ind]+=np.round(min(ml_endtime-time_stamp[ind],300)/60*45e3/5,3)
                    if ind<53:
                        ind+=1
                    else:
                        break
        VS=pd.read_csv(f"{folder}VS.csv")
        VS=VS[VS.VSCAT=="b'VERILY HEART RATE'"]
        VS=VS[(VS.VSDTC<=s_time+30*60) & (VS.VSDTC>=s_time-240*60)]
        #print(len(VS))
        if len(VS)<1620:
            continue
        hr=np.zeros(54)
        for i in range(54):
            if i<53:
                hr_range=VS[(VS.VSDTC>=time_stamp[i]) & (VS.VSDTC<time_stamp[i+1])]
            else:
                hr_range=VS[(VS.VSDTC>=time_stamp[i])]
            if hr_range.empty:
                continue
            hr[i]=np.mean(hr_range.VSSTRESN.to_numpy()[:30])

        if np.min(hr)<1:
            VS=VS[VS.VSCAT=="b'POLAR HEART RATE'"]
            VS=VS[(VS.VSDTC<=s_time+30*60) & (VS.VSDTC>=s_time-240*60)]
            for i in range(54):
                hr_range=VS[VS.VSDTC>=time_stamp[i]]
                if hr_range.empty or hr[i]>0:
                    continue
                hr[i]=np.mean(hr_range.VSSTRESN.to_numpy()[:30])
            if np.min(hr)<1:
                continue
        print(ind)
        ##insulin
        DX=pd.read_csv(f"{folder}DX.csv")
        FACM=pd.read_csv(f"{folder}FACM.csv")
        basalflow=FACM[FACM.FATEST=="b'BASAL FLOW RATE'"]
        bolus=FACM[FACM.FATEST=="b'BOLUS INSULIN'"]
        bolus=bolus[(bolus.FADTC<=s_time+30*60) & (bolus.FADTC>=s_time-240*60)]
        ba=np.zeros(54)
        for i in range(53):
            bf1=basalflow[(basalflow.FADTC<=time_stamp[i])]
            bf2=basalflow[(basalflow.FADTC>time_stamp[i])&(basalflow.FADTC<time_stamp[i+1])]
            bf2_times=bf2.FADTC.to_numpy()
            bf2_val=np.nan_to_num(bf2.FASTRESN.to_numpy())
            bf_val=np.concatenate([np.nan_to_num(bf1.FASTRESN.to_numpy())[-1:],bf2_val])
            bf_times=np.concatenate([[time_stamp[i]],bf2_times,[time_stamp[i+1]]])
            for j in range(len(bf_val)):
                ba[i]+=bf_val[j]/60*(bf_times[j+1]-bf_times[j])/(time_stamp[i+1]-time_stamp[i])
        #print(ba)
        ba[-1]=ba[-2]
        if np.mean(ba)<=0:
            continue
        #print(bolus)
        #print(time_stamp)
        for null,bolus_dose in bolus.iterrows():
            dose_time=bolus_dose.FADTC
            ind=np.searchsorted(time_stamp,dose_time)
            if (ind==54):
                continue
            dose_endtime=bolus_dose.FADTC+bolus_dose.FASTRESN/1.5*60
            #print(time_stamp[ind-1])
            #print(time_stamp[ind])
            #print(dose_time)
            #print(dose_endtime)
            if dose_endtime<time_stamp[ind]:
                ba[ind-1]+=np.round(bolus_dose.FASTRESN/5,3)
            else:
                ba[ind-1]+=np.round((time_stamp[ind]-dose_time)/60*1.5/5,3)
                while time_stamp[ind]<=dose_endtime:
                    ba[ind]+=np.round(min(dose_endtime-time_stamp[ind],300)/60*1.5/5,3)
                    ind+=1
                    if (ind==54):
                        break
        #print(ba)
        

        ##step count
        ST=pd.read_csv(f"{folder}Step_Count.csv")
        ST=ST[(ST.FADTC<=s_time+30*60-1.9e9) & (ST.FADTC>=s_time-240*60-1.9e9)]
        sc=np.zeros(54)
        for i in range(54):
            sc_range=ST[ST.FADTC>=time_stamp[i]-1.9e9]
            if (sc_range.empty):
                continue
            sc[i]=np.mean(sc_range.FASTRESN.to_numpy()[:30])
        #print(s_time)
        #print(cgm.shape)
        time_series=np.nan_to_num(np.stack([cgm,ba,meal,hr,sc],axis=1))
        
        
        if plot:
            size=15
            fig,ax=plt.subplots(8,1,figsize=(12,8),sharex=True)
            ax[0].scatter(time_stamp-s_time,cgm,c='red',s=size,label="raw CGM in mg/dl")
            #ax[0].plot(time_stamp-s_time,cgm,label="processed CGM in mg/dl")
            ax[0].set_xticks(time_stamp-s_time,minor=True)
            ax[0].set_xticks([(i*30-240)*60 for i in range(10)],[i*30-240 for i in range(10)],minor=False)
            for i in range(8):
                ax[i].xaxis.grid(True,which='minor')
            ax[0].set_ylim([30,400])
            ax[1].set_ylim([0,5])
            ax[2].set_ylim([0,6])
            ax[3].set_ylim([0,2])
            ax[4].set_ylim([0,200])
            ax[5].set_ylim([0,45000])
            ax[6].set_ylim([50,220])
            ax[7].set_ylim([0,60])
            plt.xlabel("Time from exercise onset/min")
            print(f"processing pat {pat} exe {s_time}") 
            basal_time=basalflow.FADTC.to_numpy()
            basal_flow=basalflow.FASTRESN.to_numpy()
            t0=s_time-240*60
            tN=s_time+30*60
            ba_ind1=np.searchsorted(basal_time,t0)
            ba_ind2=np.searchsorted(basal_time,tN)
            basal_times=np.concatenate([[-240*60.0],basal_time[ba_ind1:ba_ind2]-s_time,[1800.0]])
            basal_vals=np.concatenate([basal_flow[ba_ind1-1:ba_ind2],basal_flow[ba_ind2-1:ba_ind2]])
            basal_vals=np.nan_to_num(basal_vals)

            ax[1].scatter(basal_times[:-1],basal_vals[:-1],\
                        color='red',s=size,label="Raw basal flow rate in U/hour")
            ax[1].step(basal_times,basal_vals,where='post',\
                       label="Interpolated basal flow rate in U/hour")


            bolus_times=bolus.FADTC.to_numpy()-s_time
            bolus_vals=bolus.FASTRESN.to_numpy()
            ax[2].scatter(bolus_times,bolus_vals,\
                        color='red',s=size,label='Raw Bolus in U')
            #print(bolus_times,bolus_vals)
            bolus_times,bolus_vals=process_bolus(bolus_times,bolus_vals,1.5)
            #print(bolus_times,bolus_vals)
            bolus_times,bolus_vals=step_bolus(bolus_times,bolus_vals,1.5)
            
            ax[2].step(bolus_times,bolus_vals,where='post',label='Interpolated Bolus Rate in U/min')
            iir_t,iir_v=add_step(basal_times,basal_vals/60,bolus_times,bolus_vals)
            
            ax[3].step(iir_t,iir_v,where='post',label="Interpolated Insulin Rate in U/min")
            ax[3].step(time_stamp-s_time,ba,where="post",c='black',label="Smoothed Insulin Rate in U/min")
            ml_times=np.array([0.0])
            ml_vals=np.array([0.0])
            res_times=np.array([0.0])
            res_vals=np.array([0.0])
            if (not ml.empty):
                ml_times=ml.FADTC.to_numpy()-s_time
                ml_vals=[float(item[2:-1]) for item in ml.FAORRES]
                ax[4].scatter(ml_times,ml_vals,\
                             color='red', s=size,label="Raw Meal+Res_Carb in g")
            if (not res_carb.empty):
                res_times=res_carb.MLDTC.to_numpy()-s_time
                res_vals=res_carb.MLDOSE.to_numpy()
                ax[4].scatter(res_times,res_vals,\
                             color='red',s=size,label="Raw Meal+Res_Carb in g")
            #print(f"m_times{ml_times}")
            #print(f"m_vals{ml_vals}")
            ml_times,ml_vals=process_bolus(ml_times,ml_vals,45.0)
            #print(f"m_times{ml_times}")
            #print(f"m_vals{ml_vals}")
            ml_times,ml_vals=step_bolus(ml_times,ml_vals,45.0)
            #print(f"m_times{ml_times}")
            #print(f"m_vals{ml_vals}")
            res_times,res_vals=process_bolus(res_times,res_vals,45.0)
            res_times,res_vals=step_bolus(res_times,res_vals,45.0)
            carb_times,carb_vals=add_step(ml_times,ml_vals,res_times,res_vals)
            ax[5].step(carb_times,carb_vals*1e3,where='post',label="Interpolated meal rate in mg/min")
            ax[5].step(time_stamp-s_time,meal,where="post",c='black',label="Smoothed meal rate in mg/min")
            
            #print(meal)
            #print(carb_times)
            #print(carb_vals)
            vs_time=VS.VSDTC.to_numpy()
            vs_val=VS.VSSTRESN.to_numpy()
            ax[6].scatter(vs_time-s_time,vs_val,s=0.3,c='red',alpha=0.7,label="Raw Verily HR in bpm")
            ax[6].step(time_stamp-s_time,hr,where="post",label="Processed HR in bpm")
            sc_time=ST.FADTC.to_numpy()
            sc_val=ST.FASTRESN.to_numpy()
            ax[7].scatter(sc_time-s_time+1.9e9,sc_val,s=0.3,c='red',alpha=0.7,label="Raw Step Count")
            ax[7].step(time_stamp-s_time,sc,where="post",label="Processed Step Count")
            lgd=[]
            for i in range(8):
                lgdi=ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
                lgd.append(lgdi)
            fig.savefig(f"fixrange_pat_plots/{pat}_exe_{s_time}.png",\
                        bbox_extra_artists=lgd, bbox_inches='tight')
            plt.close(fig)
        
        
        #modification
        #little more/more carb on board at begining
        #little/more insulin on board at beginning of exercise
        #increase/decrease insulin during exercise
        #change exercise heart rate/step count
        series1=np.array(time_series)
        series2=np.array(time_series)
        series3=np.array(time_series)
        if dice1<0.25:
            #little more/more carb on board at begining
            series2[-7,2]+=50*200
            series3[-7,2]+=100*200
            if dice2<0.5:
                full_series=np.stack([time_series,series1,series2,series3],axis=0)
                rank=np.array([0,0,1])
            else:
                full_series=np.stack([time_series,series1,series3,series2],axis=0)
                rank=np.array([0,1,0])
        elif dice1<0.5:
            #add insulin/carb on board at beginning of exercise
            series2[-7,2]+=50*200
            series3[-7,1]+=10/5
            if dice2<0.5:
                full_series=np.stack([time_series,series1,series2,series3],axis=0)
                rank=np.array([0,1,0])
            else:
                full_series=np.stack([time_series,series1,series3,series2],axis=0)
                rank=np.array([0,0,1])
        elif dice1<0.75:
            #add little/more insulin during exercise
            series2[-7:,1]+=2.5/5
            series3[-7:,1]+=5/5
            if dice2<0.5:
                full_series=np.stack([time_series,series1,series2,series3],axis=0)
                rank=np.array([1,0,0])
            else:
                full_series=np.stack([time_series,series1,series3,series2],axis=0)
                rank=np.array([1,0,0])
        else:
            ##change exercise heart rate/step count
            series1[-7:,-2]=np.array([80,90,100,110,120,130,120])
            series2[-7:,-2]=np.array([80,170,80,170,80,170,80])
            series3[-7:,-2]=np.array([160,170,180,170,160,180,160])
            if dice2<0.5:
                full_series=np.stack([time_series,series1,series2,series3],axis=0)
                rank=np.array([0,0,1])
            else:
                full_series=np.stack([time_series,series3,series2,series1],axis=0)
                rank=np.array([1,0,0])
        train_cases.append(full_series)
        #train_cov.append(cov)
        train_ranks.append(rank)
        pat_count+=1
train_cases=np.stack(train_cases,axis=0)
#train_cov=np.stack(train_cov,axis=0)
train_ranks=np.stack(train_ranks,axis=0)
print(train_cases.shape)
#print(train_cov.shape)
print(train_ranks.shape)
np.save("new_icml_cases.npy",train_cases)
np.save("new_icml_ranks.npy",train_ranks)
