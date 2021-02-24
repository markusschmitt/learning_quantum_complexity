import numpy as np # generic math functions
from scipy.interpolate import splrep, splev

from quspin.operators import * # Hamiltonians and operators
from quspin.basis import spin_basis_1d, boson_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import pickle
import gzip
import time
import os
import sys

def generate_op(L,sup,op,verbose=False): 
 
    ##### construct basis
    basis=spin_basis_1d(L=L,a=1,kblock=0)
    dynamic=[]

    # Hamiltonian
    if sup==1:
        term_1=[[1/L,i] for i in range(L)] 
    elif sup==2:
        term_1=[[1/L,i,(i+1)%L] for i in range(L)] #PBC
    elif sup==3:
        term_1=[[1/L,i,(i+1)%L,(i+2)%L] for i in range(L)] #PBC    
        
    #print(term_1)
    # Assemble "Hamiltonian" = log(rho) operator
    static=[[op,term_1]]

    return hamiltonian(static,dynamic,basis=basis,dtype=np.complex64,check_herm=verbose,check_symm=verbose)

def H_term(L,pref,sup):
    if sup==1:
        return [[pref,i] for i in range(L)]
    elif sup==2:
        return [[pref,i,(i+1)%L] for i in range(L)]
    elif sup==3:
        return [[pref,i,(i+1)%L,(i+2)%L] for i in range(L)]
    elif sup==4:
        return [[pref,i,(i+1)%L,(i+2)%L,(i+3)%L] for i in range(L)]
    else:
        print("too large support")

# density matrix containing all H candidate terms
def generate_gge(L,g_vec,op_vec_form,op_vec_sup,verbose=False): 
 
    ##### construct basis
    basis=spin_basis_1d(L=L,a=1,kblock=0)
    dynamic=[]
    
    # Assemble "Hamiltonian" = log(rho) operator
    static=[]
    for i in range(len(op_vec_form)):
        # generate term in the H
        term=H_term(L,g_vec[i],op_vec_sup[i])  
        static.append([op_vec_form[i],term])

    H=hamiltonian(static,dynamic,basis=basis,dtype=np.complex64,check_herm=verbose,check_symm=verbose)
    H_eigval, H_eigvec = H.eigh()
        
    rho = H_eigvec.dot(np.diag(np.exp(H_eigval)).dot(np.transpose(np.conj(H_eigvec))))
    
    return rho/np.trace(rho)


# Jacobi matrix
def jacobi_exact(g_vec, op_vec_form, op_vec_sup, c, L):
    
    J=np.zeros((len(op_vec_form),len(op_vec_form)))
    
    for j in range(len(op_vec_form)):
        op=generate_op(L,op_vec_sup[j],op_vec_form[j])
        
        for i in range(len(g_vec)):
            g=g_vec[i]
            dg=abs(g)/c
            x=np.linspace(g-dg, g+dg, num=11)
            y=np.zeros(len(x))

            for k in range(len(y)): 
                g_vec_d=g_vec.copy()
                g_vec_d[i]=x[k]  
                rho_m=generate_gge(L,g_vec_d,op_vec_form,op_vec_sup)
                y[k]=np.real(np.trace(op.static.dot(rho_m)))

            f = splrep(x,y,k=5,s=3)
            dydx=splev(x,f,der=1)  
            
            J[j,i]=dydx[6]
    
    return J

#Newtorn's method
def newton_exact(g_vec, op_vec_form, op_vec_sup, vec_op_num, c, L):
    
    rho=generate_gge(L,g_vec,op_vec_form,op_vec_sup)
    vec_op=[]
    for i in range(len(op_vec_form)):
        op=generate_op(L,op_vec_sup[i],op_vec_form[i])
        vec_op.append(np.real(np.trace(op.static.dot(rho))))
    
    J_mat_I=np.linalg.inv(jacobi_exact(g_vec, op_vec_form, op_vec_sup, c, L))
    
    return np.subtract(g_vec,J_mat_I.dot(np.subtract(vec_op,vec_op_num))), np.subtract(vec_op,vec_op_num)


# Do analysis on one example
data_raw=np.loadtxt("./../data/training_data/gge/L=12_charges=1000_support=3_J=1_g=0.6_numData=2000.dat")[::2]

beta=0.1;
g_vec=[beta,beta*0.2,beta*0.4]
op_vec_form=["zz","x","zxz"]
op_vec_sup=[2,1,3]
c=10 
L=12 #system size
ind=14 #index of realizations

vec_op_num=[2*data_raw[ind,3],2*data_raw[ind,47],2*data_raw[ind,8]]

err=1; i=0
print("Progress in H couplings")
while i<100 and err>1e-6:
    tmp=newton_exact(g_vec, op_vec_form, op_vec_sup, vec_op_num, c, L)
    g_vec=tmp[0]
    err=np.linalg.norm(tmp[1])
    i=i+1
    print([g_vec, err])
g_vec

print()
print(["hx/J:",g_vec[1]/g_vec[0]])