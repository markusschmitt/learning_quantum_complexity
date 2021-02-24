using Pkg
Pkg.activate(".")

using PyPlot
using DelimitedFiles
using StatsBase, Optim

"""
    Euc_dist(xi,xj)
Euclidean distance between vector xi and xj
"""
Euc_dist(xi,xj) = sqrt(sum((xi.-xj).^2))

"""
    Id(nB,data)
returns ln(mu), ln(1-P(mu))
nB: sets the width of bins, 
data: contains rows of vectors corresponding to the input 
or the latent representation of each data set of expectation values
"""
function Id(nB,data)
    data=readdlm(path*dir*file, Float64)
    Nreal=size(data,1)
    
    mu=[]
    for i in 1:Nreal
        xi=data[i,:] 
        dst=[]
        for j in 1:Nreal
            xj=data[j,:]
            (i!=j) ? push!(dst,Euc_dist(xi,xj)) : nothing
        end
        sort_dst=sort(dst)
        push!(mu,Float64(sort_dst[2]/sort_dst[1]))
    end
    mu=convert(Array{Float64,1}, mu);

    histo=fit(Histogram, mu, nbins=Int(floor(maximum(mu)/nB)))
    PMu=histo.weights
    PMu=PMu/sum(PMu)
    Pcum = [ sum(view(PMu,1:n)) for n in 1:size(PMu,1)-1]
    dx=histo.edges[1][2]-histo.edges[1][1]
    xdata=log.(0.5*dx.+[histo.edges[1][i] for i in 1:size(histo.edges[1],1)-2])
    ydata=log.(1.0 .-Pcum)
    
    return xdata, ydata
end

ssqr(x,y,p,model) = sum((y .- model(x,p)).^2)
@. model(x,p) = p[1] + p[2]*x

"""
    fit_Id(x,y; p0=ones(2))
    x = ln(mu)
    y = ln(1-P(mu))
fits Id from relation ln(1-P(mu)) = -Id * ln(mu)
"""
function fit_Id(x,y; p0=ones(2))
    lb = [0.]
    ub = [10.]

    df = TwiceDifferentiable(p-> ssqr(x,y,p,model), p0; autodiff=:forward)
    dfc = TwiceDifferentiableConstraints(lb,ub)

    res = optimize(df,dfc,p0,IPNewton())
    return res.minimizer
end


fig, ax=subplots(figsize=(7,5))

path="/Users/zalalenarcic/Projects/ML_GGE/data/data_for_article/latent_values/Fig1"
#path=pwd()
dir="/ed_p=none_N=12_betas=1100_support=3_J=1_g=0.6_numData=2000/"
file="latent-4_epochs-250000.dat_latent_values.txt" 

data=readdlm(path*dir*file, Float64)

nB=0.03;
tmp=Id(nB,data)

# fit slope to get Id; discard large mu > xmax
xmax = 0.7
datax=tmp[1]
datax = datax[datax .< xmax]
datay=view(tmp[2],1:size(datax,1))
sl=fit_Id(datax,datay)
println("Id=$(-sl[2])")

ax.plot(tmp[1], tmp[2],label=L"N_C=2, I_d=2.0",color="black")

ax.set_ylim([-2.49,0])
ax.set_xlim([0,1])
ax.set_xlabel(L"\ln(\mu)",fontsize=16)
ax.set_ylabel(L"\ln(1-P(\mu))",fontsize=16)