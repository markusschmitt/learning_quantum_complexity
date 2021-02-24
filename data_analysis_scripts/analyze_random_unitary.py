import numpy as np
import sys
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import sys
sys.path.append(sys.path[0]+"/..")

dn="./data/results/random_unitary/L=20/"
tdirs=glob.glob(dn+"/obs_t*")

# Collect data
res={}
times=[]
allErrors=[]
for td in tdirs:
    if os.path.isdir(td):

        T = float(td.split("=")[-1])
        times.append(T)

        fs = glob.glob(td+"/latent-*00.dat")

        errs = []
        for f in fs:
            L = int(f.split("latent-")[-1].split("_")[0])
            if L<20:
                data=np.loadtxt(f)
                minidx = np.argmin(data[:,2])
                errs.append([float(T), L, data[minidx,2]]) 

        errs=np.array(errs)
        errs=errs[np.argsort(errs[:,1])]
        allErrors.append(errs[:,2])
        res[str(T)] = errs

fitData = []
for time, data in res.items():

    fitData.append(data)

fitData=np.array(fitData)
fitData = fitData.reshape((-1,3))


times=np.array(times)
allErrors=np.array(allErrors)
LInterp=np.arange(1,10,1)
tInterp=np.arange(0.1,49,1)

X,Y=np.meshgrid(LInterp, tInterp)
from scipy.interpolate import griddata
#from scipy.interpolate import interp2d
from scipy.interpolate import bisplrep, bisplev

def mylog(x, r=1.0):

    if x>r:
        return r-np.log(r)+np.log(x)

    return x

R=0.05
idx=np.where(fitData[:,0]>R)
fitData[idx[0],0]=R-np.log(R)+np.log(fitData[idx[0],0])

sc=plt.scatter(fitData[:,0],fitData[:,1],c=np.log(fitData[:,2])/np.log(10.), cmap='gist_heat_r')
cbar=plt.colorbar(sc)
plt.xlim(-.1,5.1)
plt.clim(-6,-1.9)
plt.xticks(np.array([mylog(0.,R),mylog(0.1,R),mylog(0.2,R),mylog(0.5,R),mylog(1.,R),mylog(2.,R),mylog(5.,R),mylog(50.,R)]), ['0', '0.1', '0.2', '0.5', '1', '2', '5', '50'])
plt.yticks(np.array([1,2,3,4,6,8,10]))
plt.ylabel('# latent')
plt.xlabel('time')
cbar.set_label('log. test error')
plt.tight_layout()
plt.savefig(dn+"/rnd_unitaries_complexity_vs_time.pdf")

plt.close()

ptstyle={'edgecolors':'white', 'linewidths':0.5}

fig, ax = plt.subplots(figsize=(5,3.25))

a=bisplrep(fitData[:,0],fitData[:,1], np.log(fitData[:,2])/np.log(10.),kx=3,ky=5,s=3)
#errInterp = interp2d(fitData[:,0],fitData[:,1], np.log(fitData[:,2])/np.log(10.), kind='linear')
Ls=np.arange(0,10.5,1)
ts=np.arange(0,mylog(50,R),.01)
#Cs=errInterp(ts,Ls)
Cs=np.transpose(np.array(bisplev(ts,Ls,a)))

def cdef(x):
    return [1,1,1-x/256,1]

cmap='Reds'
cmap=cm.get_cmap(cmap,256)
nsteps=8
discr_cmap=ListedColormap([cmap(i*256//nsteps) for i in range(nsteps)])
cax=ax.imshow(Cs, origin='lower', cmap=discr_cmap, extent=[min(ts),max(ts),min(Ls),max(Ls)], aspect='auto', interpolation='bicubic')
ax.set_autoscale_on(False)
sc=plt.scatter(fitData[:,0],fitData[:,1],c=np.log(fitData[:,2])/np.log(10.), cmap=cmap, **ptstyle)
cbar=fig.colorbar(sc,ticks=[-7,-6,-5,-4,-3,-2])
cbar.ax.set_yticklabels([r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])
ax.set_xlim(-0.1,mylog(50.,R)+0.1)
ax.set_ylim(-0.2,10.2)
ax.axes.set_xticks(np.array([mylog(0.,R),mylog(0.1,R),mylog(0.2,R),mylog(0.5,R),mylog(1.,R),mylog(2.,R),mylog(5.,R),mylog(50.,R)]))
ax.axes.set_xticklabels([r'$0$', r'$1$', r'$2$', r'$5$', r'$10$', r'$20$', r'$50$', r'$500$'])
ax.axes.set_yticks(np.array([0,1,2,3,4,6,8,10]))
ax.set_ylabel(r'Number of latent variables $N_L$')
ax.set_xlabel(r'Time')
cbar.set_label(r'Test error')

fig.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.14, right=0.97, top=0.98, wspace=0, hspace=0)

fig.savefig("figures/fig3.pdf")
