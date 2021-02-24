import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append(sys.path[0]+"/..")

from sklearn.manifold import TSNE

import torch
from torch import optim

def colorbar(mappable, dummy=False):
    # from https://joseph-long.com/writing/colorbars/
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if dummy:
        cax.set_axis_off()
        return None
    cbar = fig.colorbar(mappable, cax=cax, shrink=0.98)
    plt.sca(last_axes)
    return cbar

def center(a,b):
    X=a-0.5*(np.max(a)+np.min(a))
    Y=b-0.5*(np.max(b)+np.min(b))
    return X,Y

Fs=["./data/results/gge/L=12_charges=1000_support=3_J=1_g=0.6_numData=2000/",
    "./data/results/gge/L=12_charges=1100_support=3_J=1_g=0.6_numData=2000/",
    "./data/results/gge/L=12_charges=1110_support=3_J=1_g=0.6_numData=2000/"]


fig = plt.figure(figsize=(5.25, 6.5))
fig.text(0.04,0.955,r'a)')
fig.text(0.4,0.955,r'b)')
fig.text(0.04,0.435,r'c)')
outer = gridspec.GridSpec(2,2)

#####
# Plot panel A

# Load carton of network
img = mpimg.imread('./figures/Fig1a.png')
ax = plt.Subplot(fig, outer[0])
ax.imshow(img, interpolation='none')
ax.annotate('', xy=(-0.05, 0.1), xycoords='axes fraction', xytext=(-0.05, 0.9), 
            arrowprops=dict(arrowstyle="->", color='black'))
fig.text(0.05,0.72,r'data flow', rotation=90)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
fig.add_subplot(ax)

# Panel A done
#####


#####
# Plot panel B

ax = plt.Subplot(fig, outer[1])

colors=['black', 'darkred', 'orangered']
labels=[r'$N_C=1$', r'$N_C=2$', r'$N_C=3$']
j=0
fitRange1=[2,3,4]
for dn in Fs:
    fs=glob.glob(dn+"/latent-*_epochs-250000.dat")
    Ls=[]
    errs=[]
    for f in fs:
        L = int(f.split("latent-")[1].split("_epochs")[0])
        if L<10:
            Ls.append(L)
            data = np.loadtxt(f)
            err = np.min(data[:,2])
            errs.append(err)

    Ls=np.array(Ls)
    srtIdx=np.argsort(Ls)
    errs=np.array(errs)

    ax.semilogy(Ls[srtIdx],errs[srtIdx], 'o', c=colors[j], label=labels[j])

    a,b = np.polyfit(Ls[srtIdx][:fitRange1[j]],np.log(errs[srtIdx][:fitRange1[j]]),1)
    l=np.array([0,fitRange1[j]-0.5])
    ax.semilogy(l,np.exp(a*l+b), '--', c=colors[j])
    a,b = np.polyfit(Ls[srtIdx][fitRange1[j]-1:],np.log(errs[srtIdx][fitRange1[j]-1:]),1)
    l=np.array([fitRange1[j]-1.5,6])
    ax.semilogy(l,np.exp(a*l+b), '--', c=colors[j])
    j=j+1

ax.set_ylim(1e-10,1.2e-2)
ax.set_xlabel(r'Number of latent variables $N_L$')
ax.set_ylabel(r'Test error')
ax.set_xticks([0,1,2,3,4,5,6])
ax.legend()
fig.add_subplot(ax)

# Panel B done
#####


#####
# Plot panel C

ptstyle={'cmap':'RdGy', 'edgecolors':'black', 'linewidths':0.02, 's':7}

inner = gridspec.GridSpecFromSubplotSpec(3, 4,
                subplot_spec=outer[2:], wspace=-0.5, hspace=0.075)

pltIdx=0

for dn in Fs:
    trainDataFileName = dn.split("/")[-2]+".dat"
    trainData = np.loadtxt("./data/training_data/gge/"+trainDataFileName)[::2]

    means=np.mean(trainData, axis=0)

    # Need factors of two here, because of superfluous factor 1/2 in generate_ed_data.py 
    energies = 2*(trainData[:,3] + 0.6*trainData[:,47])
    C1 = 2*(trainData[:,19] - trainData[:,7])
    C2 = 2*(trainData[:,8] - 0.6 * (trainData[:,43]+trainData[:,23]) - trainData[:,47])
    C3 = 2*(trainData[:,24] - trainData[:,9])

    fs=glob.glob(dn+"/latent-*_epochs-250000.dat")
    Ls=[]
    errs=[]
    for f in fs:
        L = int(f.split("latent-")[1].split("_epochs")[0])
        
        net = torch.load(f+"_min_loss.net.pkl")         
        net.eval()
        _, latentValues = net(torch.as_tensor(trainData, dtype=torch.float32))
        latentValues = latentValues.detach().numpy()

        np.savetxt(f+"_latent_values.txt", latentValues)

        # We shoe t-SNEs for 4 latent variables
        if L==4:

            # Get t-SNE
            tsne=np.array(TSNE(n_components=2, perplexity=35,init='random',random_state=213).fit_transform(latentValues))

            X,Y = center(np.array(tsne[:,0]),np.array(tsne[:,1]))
            ax1 = plt.Subplot(fig, inner[pltIdx])
            pltIdx+=1
            sc1=ax1.scatter(X,Y, c=energies, **ptstyle)
            sc1.set_clim(-1.1,1.1)
            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)
            fig.add_subplot(ax1)
            
            ax2 = plt.Subplot(fig, inner[pltIdx])
            pltIdx+=1
            sc2=ax2.scatter(X,Y, c=C1, **ptstyle)
            sc2.set_clim(-1.1,1.1)
            ax2.axes.xaxis.set_visible(False)
            ax2.axes.yaxis.set_visible(False)
            fig.add_subplot(ax2)
            
            ax3 = plt.Subplot(fig, inner[pltIdx])
            pltIdx+=1
            sc3=ax3.scatter(X,Y, c=C2, **ptstyle)
            sc3.set_clim(-1.1,1.1)
            ax3.axes.xaxis.set_visible(False)
            ax3.axes.yaxis.set_visible(False)
            fig.add_subplot(ax3)
            
            ax4 = plt.Subplot(fig, inner[pltIdx])
            pltIdx+=1
            sc4=ax4.scatter(X,Y, c=C3, **ptstyle)
            sc4.set_clim(-1.1,1.1)
            ax4.axes.xaxis.set_visible(False)
            ax4.axes.yaxis.set_visible(False)
            fig.add_subplot(ax4)

            # Make square plots
            if np.max(np.abs(X)) > np.max(np.abs(Y)):
                ax1.set_ylim(ax1.get_xlim())
                ax2.set_ylim(ax1.get_xlim())
                ax3.set_ylim(ax1.get_xlim())
                ax4.set_ylim(ax1.get_xlim())
            else:
                ax1.set_xlim(ax1.get_ylim())
                ax2.set_xlim(ax1.get_ylim())
                ax3.set_xlim(ax1.get_ylim())
                ax4.set_xlim(ax1.get_ylim())
            ax1.set_aspect('equal',adjustable='box')
            ax2.set_aspect('equal',adjustable='box')
            ax3.set_aspect('equal',adjustable='box')
            ax4.set_aspect('equal',adjustable='box')
            
            if pltIdx%4==0:
                dummy=False
                if pltIdx==4 or pltIdx==12:
                    dummy=True
                colorbar(sc4,dummy)


fig.tight_layout()
fig.text(0.2,0.008,r'$C_0$')
fig.text(0.39,0.008,r'$C_1$')
fig.text(0.58,0.008,r'$C_2$')
fig.text(0.76,0.008,r'$C_3$')
fig.text(0.09,0.355,r'$N_C=1$', rotation=90)
fig.text(0.09,0.21,r'$N_C=2$', rotation=90)
fig.text(0.09,0.07,r'$N_C=3$', rotation=90)

# Panel C done
#####

fig.savefig("figures/fig1.pdf")
