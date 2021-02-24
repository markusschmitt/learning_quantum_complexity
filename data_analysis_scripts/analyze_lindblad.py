import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib
import matplotlib.gridspec as gridspec

import sys
sys.path.append(sys.path[0]+"/..")

import torch
from sklearn.manifold import TSNE

def colorbar(mappable):
    # from https://joseph-long.com/writing/colorbars/
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, ticks=[-1,0,1])
    cax.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
    plt.sca(last_axes)
    return cbar

F=["data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.576_hz_0.487_chi_100_eps_0.001/",
    "data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.576_hz_0.487_chi_100_eps_0.2/",
    "data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.576_hz_0.487_chi_100_eps_1.0/"]

F2=["data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.676_hz_0.0_chi_100_eps_0.005/",
    "data/results/lLindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.676_hz_0.0_chi_100_eps_0.2/",
    "data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.676_hz_0.0_chi_100_eps_1.0/"]

matplotlib.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(6.5, 5))
outer = gridspec.GridSpec(2, 2)
outer.update(right=0.99,top=0.98, hspace=0.1,wspace=0.05)

fig.text(0.01,0.96,r'a)')
fig.text(0.575,0.96,r'b)')
fig.text(0.01,0.5,r'c)')
fig.text(0.575,0.5,r'd)')

ax2 = fig.add_subplot(outer[2])
ax1 = fig.add_subplot(outer[0], sharex=ax2)
plt.setp(ax1.get_xticklabels(), visible=False)

#########
# Panel A

eps=[0.001,0.2,1.0]
i=0
colors=['black', 'firebrick', 'lightcoral']
for d in F:
    fs = glob.glob(d+"latent-*0.dat")
    errs=[]
    ls=[]
    for f in fs:
        l = int(f.split("latent-")[1].split("_")[0])
        data=np.loadtxt(f)
        ls.append(l)
        errs.append(np.min(data[:,2]))

    ls=np.array(ls)
    srtIdx=np.argsort(ls)
    errs=np.array(errs)
    ax1.semilogy(ls[srtIdx], errs[srtIdx], '-o', c=colors[i], label=r'$\epsilon='+str(eps[i])+r'$')

    i=i+1
ax1.set_ylabel(r'Test error')
ax1.legend()
fig.add_subplot(ax1)

# Panel A done
#########

#########
# Panel B

eps=[0.005,0.2,1.0]
i=0
colors=['darkred','firebrick', 'lightcoral']
for d in F2:
    fs = glob.glob(d+"latent-*0.dat")
    errs=[]
    ls=[]
    for f in fs:
        l = int(f.split("latent-")[1].split("_")[0])
        data=np.loadtxt(f)
        ls.append(l)
        errs.append(np.min(data[:,2]))

    ls=np.array(ls)
    srtIdx=np.argsort(ls)
    errs=np.array(errs)
    ax2.semilogy(ls[srtIdx], errs[srtIdx], '-o', c=colors[i], label=r'$\epsilon='+str(eps[i])+r'$')

    i=i+1
ax2.set_xlabel(r'Number of latent variables $N_L$')
ax2.set_ylabel(r'Test error')
ax2.legend()
fig.add_subplot(ax2)

# Panel B done
#########


#########################
### t-SNEs
#########################

ax11 = plt.Subplot(fig, outer[1])

ptstyle={'cmap':'RdGy', 'edgecolors':'black', 'linewidths':0.02, 's':10}

d="data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.576_hz_0.487_chi_100_eps_0.001/"

fn=d.split("lindblad/")[1][:-1]

hx=2*float(fn.split("hx_")[1].split("_hz")[0])
hz=2*float(fn.split("hz_")[1].split("_")[0])

trainData=np.loadtxt("data/training_data/lindblad/"+fn+".dat")
energy=(hx*trainData[:,0]+hz*trainData[:,2])+trainData[:,11]
    
net=torch.load(d+"/latent-5_epochs-250000.dat_min_loss.net.pkl")
net.eval()
_, latentValues = net(torch.as_tensor(trainData, dtype=torch.float32))
latentValues = latentValues.detach().numpy()

# store latent values
np.savetxt(d+"latent-5_epochs-250000.dat_latent_values.txt", latentValues)

# Get t-SNE
tsne=np.array(TSNE(n_components=2, perplexity=47,init='random',random_state=213).fit_transform(latentValues))

X=np.array(tsne[:,0])
X=X-0.5*(np.max(X)+np.min(X))
Y=np.array(tsne[:,1])
Y=Y-0.5*(np.max(Y)+np.min(Y))

sc1=ax11.scatter(X, Y, c=energy, **ptstyle)
ax11.axes.xaxis.set_visible(False)
ax11.axes.yaxis.set_visible(False)
if np.max(np.abs(X)) > np.max(np.abs(Y)):
    ax11.set_ylim(ax11.get_xlim())
else:
    ax11.set_xlim(ax11.get_ylim())
ax11.set_aspect('equal',adjustable='box')
ax11.text(0.05,0.9,r'$C_0$',transform=ax11.transAxes)

sc1.set_clim(-1.1,1.1)
colorbar(sc1)

fig.add_subplot(ax11)

# Panel C done
#########

#########
# Panel D

inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                subplot_spec=outer[3], wspace=-0.35, hspace=0.05)

ax21 = plt.Subplot(fig, inner[0])
ax22 = plt.Subplot(fig, inner[1])
ax23 = plt.Subplot(fig, inner[2])
ax24 = plt.Subplot(fig, inner[3])

d="data/results/lindblad/R_op_vals_N_40_J_0.0_Jz_1.0_hx_0.676_hz_0.0_chi_100_eps_0.005/"

fn=d.split("lindblad/")[1][:-1]

hx=2*float(fn.split("hx_")[1].split("_hz")[0])
hz=2*float(fn.split("hz_")[1].split("_")[0])

trainData=np.loadtxt("data/training_data/lindblad/"+fn+".dat")
energy=(hx*trainData[:,0]+hz*trainData[:,2])+trainData[:,11]

net=torch.load(d+"/latent-5_epochs-250000.dat_min_loss.net.pkl")
net.eval()
_, latentValues = net(torch.as_tensor(trainData, dtype=torch.float32))
latentValues = latentValues.detach().numpy()

# Store latent values
np.savetxt(d+"latent-5_epochs-250000.dat_latent_values.txt", latentValues)

tsne=np.array(TSNE(n_components=2, perplexity=47,init='random',random_state=213).fit_transform(latentValues))

X=np.array(tsne[:,0])
X=X-0.5*(np.max(X)+np.min(X))
Y=np.array(tsne[:,1])
Y=Y-0.5*(np.max(Y)+np.min(Y))

sc1=ax21.scatter(X,Y, c=energy, **ptstyle)
ax21.axes.xaxis.set_visible(False)
ax21.axes.yaxis.set_visible(False)

C1=trainData[:,8]+trainData[:,10]
sc2=ax22.scatter(X,Y, c=100*C1, **ptstyle)
ax22.axes.xaxis.set_visible(False)
ax22.axes.yaxis.set_visible(False)

C2=trainData[:,41]-trainData[:,0]-hx*(trainData[:,7]+trainData[:,11])
sc3=ax23.scatter(X,Y, c=5*(energy+C2), **ptstyle)
ax23.axes.xaxis.set_visible(False)
ax23.axes.yaxis.set_visible(False)

C3=trainData[:,29]+trainData[:,40]
sc4=ax24.scatter(X,Y, c=100*C3, **ptstyle)
ax24.axes.xaxis.set_visible(False)
ax24.axes.yaxis.set_visible(False)

sc1.set_clim(-1.1,1.1)
sc2.set_clim(-1.1,1.1)
sc3.set_clim(-1.1,1.1)
sc4.set_clim(-1.1,1.1)

ax21.text(0.05,0.05,r'$C_0$',transform=ax21.transAxes)
ax22.text(0.05,0.05,r'$100\times C_1$',transform=ax22.transAxes)
ax23.text(0.05,0.05,r'$5\times(C_0+C_2)$',transform=ax23.transAxes)
ax24.text(0.05,0.05,r'$100\times C_3$',transform=ax24.transAxes)

if np.max(np.abs(X)) > np.max(np.abs(Y)):
    ax21.set_ylim(ax21.get_xlim())
    ax22.set_ylim(ax21.get_xlim())
    ax23.set_ylim(ax21.get_xlim())
    ax24.set_ylim(ax21.get_xlim())
else:
    ax21.set_xlim(ax21.get_ylim())
    ax22.set_xlim(ax21.get_ylim())
    ax23.set_xlim(ax21.get_ylim())
    ax24.set_xlim(ax21.get_ylim())
ax21.set_aspect('equal',adjustable='box')
ax22.set_aspect('equal',adjustable='box')
ax23.set_aspect('equal',adjustable='box')
ax24.set_aspect('equal',adjustable='box')
fig.add_subplot(ax21)
fig.add_subplot(ax22)
fig.add_subplot(ax23)
fig.add_subplot(ax24)

# Panel D done
#########


plt.tight_layout()
plt.savefig("figures/fig2.pdf")
