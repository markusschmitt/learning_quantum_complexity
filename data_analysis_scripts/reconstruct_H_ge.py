import sys
from sklearn.decomposition import PCA
from scipy.interpolate import splrep, splev
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.gridspec as gridspec

sys.path.append(sys.path[0]+"/..")

import torch
from sklearn.manifold import TSNE

def op_norm(i_real,i_op, data_raw):
    return data_raw[i_real,i_op]

def euc_dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)    

def find_nn_ind(i,x_data,y_data,ind_ord_nn):
    j_min=1
    dist_min=euc_dist(x_data[i],y_data[i],x_data[j_min],y_data[j_min])
    
    for j in range(len(x_data)):
        dist_tmp=euc_dist(x_data[i],y_data[i],x_data[j],y_data[j])
        if dist_tmp<dist_min and j!=i and (len(list(filter (lambda x : x == j, ind_ord_nn))) == 0):
            j_min=j
            dist_min=dist_tmp
        
    return j_min    

def colorbar(mappable):
    # from https://joseph-long.com/writing/colorbars/
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, ticks=[-1,0,1])
    plt.sca(last_axes)
    return cbar


#########################
### t-SNEs
#########################

ptstyle={'cmap':'RdGy', 'edgecolors':'black', 'linewidths':0.01, 's':40}


trainData=np.loadtxt("./../data/training_data/gge/L=12_charges=1000_support=3_J=1_g=0.6_numData=2000.dat")[::2]
energy=2*(trainData[:,3]+0.6*trainData[:,47]) 

net=torch.load("./../data/results/gge/L=12_charges=1000_support=3_J=1_g=0.6_numData=2000/latent-4_epochs-250000.dat_min_loss.net.pkl")
net.eval()
_, latentValues = net(torch.as_tensor(trainData, dtype=torch.float32))
latentValues = latentValues.detach().numpy()

#make tSNE 
tsne=np.array(TSNE(n_components=2, perplexity=35,init='random',random_state=213).fit_transform(latentValues))

# center plot
X=np.array(tsne[:,0])
X=X-0.5*(np.max(X)+np.min(X))
Y=np.array(tsne[:,1])
Y=Y-0.5*(np.max(Y)+np.min(Y))

fig, ax = plt.subplots(figsize=(3.5,3))
sc=ax.scatter(X, Y, c=energy, **ptstyle)
ax.set_aspect('equal',adjustable='box')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

if np.max(np.abs(X)) > np.max(np.abs(Y)):
    ax.set_ylim(ax.get_xlim())
else:
    ax.set_xlim(ax.get_ylim())   
      
colorbar(sc)
plt.tight_layout()


###########################################################################
### finding H candidate terms from gradient along the first pca component
###########################################################################

# order tsne data with respect to NN along the 1D manifold
x_data=tsne[:,0]
y_data=tsne[:,1]

# start with the point with minimal x coordiate (an arbitrary choice)
ind=np.argmin(x_data)
ind_ord_nn=[ind]

for j in range(len(x_data)):
    ind=find_nn_ind(ind,x_data,y_data,ind_ord_nn)
    ind_ord_nn=np.append(ind_ord_nn,ind)
    
#calculate gradient along the 1D manifold and store them in max_grad
n_pl=len(trainData[1]);
max_grad=[]
for i_op in range(n_pl):
    
    x = np.array(range(len(ind_ord_nn)-50))
    noisy_data = [trainData[ind_ord_nn[i],i_op] for i in range(len(ind_ord_nn)-50)]
    f = splrep(x,noisy_data,k=5,s=3)
    dydx=splev(x,f,der=1)
    
    max_grad=np.append(max_grad,abs(dydx.sum())/len(dydx))


# plot max_grad
fig,(ax1)=plt.subplots(1,1,sharex=True,sharey=True,figsize=(3,3))
plt.plot(max_grad,'ko-') 

#dominant terms are
#3=zz
#47=x
#8=zxz

ax1.axis('tight')
ax1.set_xlabel(r'$\vec\alpha$', fontsize=18)
ax1.set_ylabel(r'$\overline{\partial \langle O(\vec\alpha)\rangle}$', fontsize=18)
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(16)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_yticks([])

plt.show()   