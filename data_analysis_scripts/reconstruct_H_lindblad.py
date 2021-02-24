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

ptstyle={'cmap':'RdGy', 'edgecolors':'black', 'linewidths':0.07, 's':40}

trainData=np.loadtxt("./../data/training_data/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.576_hz_0.487_h_0.0_chi_100_eps_0.001.dat")
energy=2*(0.576*trainData[:,0]+0.487*trainData[:,2])+trainData[:,11]

net=torch.load("./../data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.576_hz_0.487_h_0.0_chi_100_eps_0.001/latent-5_epochs-250000.dat_min_loss.net.pkl")
net.eval()
_, latentValues = net(torch.as_tensor(trainData, dtype=torch.float32))
latentValues = latentValues.detach().numpy()

#makes tSNE and PCA analysis on it
tsne=np.array(TSNE(n_components=2, perplexity=47,init='random',random_state=213).fit_transform(latentValues))
pca = PCA(n_components=2)
pca.fit(tsne)

# center plot
X=np.array(tsne[:,0])
X=X-0.5*(np.max(X)+np.min(X))
Y=np.array(tsne[:,1])
Y=Y-0.5*(np.max(Y)+np.min(Y))

fig,ax = plt.subplots(figsize=(3.5,3))

sc=ax.scatter(X, Y, c=energy, **ptstyle)
sc.set_clim(-1.1,1.1)

ax.set_aspect('equal',adjustable='box')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

if np.max(np.abs(X)) > np.max(np.abs(Y)):
    ax.set_ylim(ax.get_xlim())
else:
    ax.set_xlim(ax.get_ylim())
    
i_comp=0; cf=13
plt.arrow(cf*pca.components_[i_comp][0], cf*pca.components_[i_comp][1], -2*cf*pca.components_[i_comp][0], -2*cf*pca.components_[i_comp][1], head_width=1, head_length=2, fc='k', ec='k')


#calculate tsne-vector projections on the first pca vector
i_comp=0
data_proj=[np.dot(pca.components_[i_comp],tsne[i]) for i in range(len(tsne))]

colorbar(sc)
plt.tight_layout()


###########################################################################
### finding H candidate terms from gradient along the first pca component
###########################################################################

n_pl=len(trainData[1]);
n_ave=30
ind_list=np.argsort(data_proj)

#calculate gradient along first PCA and store them in max_grad
max_grad=[]
for i_op in range(n_pl):
    out=[]
    for i_real in range(len(ind_list)):
        out=np.append(out,op_norm(ind_list[i_real],i_op,trainData))
        
    conv_out=np.convolve(out, np.ones((n_ave,))/n_ave, mode='valid')
    
    x = np.array(range(len(conv_out)))
    noisy_data = conv_out
    f = splrep(x,noisy_data,k=5,s=3)
    dydx=splev(x,f,der=1)
    
    max_grad=np.append(max_grad,abs(dydx.sum())/len(dydx))
    
# plot max_grad        
fig,(ax1)=plt.subplots(1,1,sharex=True,sharey=True,figsize=(3,3))
plt.plot(max_grad,'ko-') 

#largest gradient show terms that are part of H, H = h_x X + h_z Z + J zz
#max_grad[0] --> x
#max_grad[2] --> z
#max_grad[11] --> zz
#see published_code/data/training_data/lindblad/S=4_observable_list.dat for
# operator ordering in the R_op_vals files

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