import numpy as np
import matplotlib.pyplot as plt
data=[]
rank=[]
for j in range(1000):
    rng=np.random.default_rng(j)
    y0=0
    dt=1e-3
    beta=10*(rng.random()+0.5)
    eps=rng.normal(0,1e-4,int(1/dt))
    A=rng.random()+1
   

    #generate X
    t=np.array([i*dt for i in range(int(1/dt))])
    x1=A*np.exp(-beta*t)
    x2=1.5*x1+eps
    y=[y0]
    for i in range(int(1/dt)-1):
        dy=dt*(-y[-1]+x1[i]-x2[i])
        y.append(y[-1]+dy)
    #fig,ax=plt.subplots(2,1,sharex=True)
    #ax[0].plot(t,y,c='black',label='y')
    #ax[1].plot(t,x1,c='r',label='x1')
    #ax[1].plot(t,x2,c='blue',label='x2')
    #plt.legend()
    #plt.show()
    series1=np.concatenate([np.expand_dims(y[0:999:10],axis=1),np.expand_dims(x1[0:999:10],axis=1),np.expand_dims(x2[0:999:10],axis=1)],axis=-1)
    series2=np.copy(series1)
    series3=np.copy(series1)
    roll=rng.random()
    if (roll<0.33):
        series2[-10:,1]+=1
        series3[-10:,1]+=2
        rank.append([0,0,1])
    elif (roll<0.66):
        series2[-10:,2]+=1
        series3[-10:,2]+=2
        rank.append([1,0,0])
    else:
        series2[-10:,1]+=1
        series3[-10:,2]+=1
        rank.append([0,1,0])
    data.append([series1,series1,series2,series3])
data=np.array(data)
rank=np.array(rank)
print(data.shape)
print(rank.shape)
np.save("synthetic_cases.npy",data)
np.save("synthetic_ranks.npy",rank)