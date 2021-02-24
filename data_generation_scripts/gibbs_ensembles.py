#
# Script to generate training data from (generalized) Gibbs states
# of the transverse-field Ising model
#
# Author: Markus Schmitt and Zala Lenarcic
# Date: Feb 2021
#

from quspin.operators import * # Hamiltonians and operators
from quspin.basis import spin_basis_1d, boson_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import pickle
import gzip
import time
import os
import sys

################
# Simulation parameters

L=12                 # System size
J=1                  # Ising coupling
h=0.0                # Longitudinal field
g=0.6                # Transverse field
numData=2000         # Number of samples
support=3            # Support of observables
chargeChoice="1000"  # Choice of charges: each character 0/1 at position j indicates whether to include charge C_j

# Output file name
fn="L="+str(L)+"_charges="+chargeChoice+"_support="+str(support)+"_J="+str(J)+"_g="+str(g)+"_numData="+str(numData)+".dat"
# Output directory
dn="./data/training_data/gge/"

################


def opname(idx):
    """Translate boson basis index to Pauli operator name
    """
    if idx=="0":
        return "", 0, 1
    if idx=="1":
        return "x", 1, 1
    if idx=="2":
        return "y", 1, 1
    if idx=="3":
        return "z", 1, 1

    return "", 0, 0


def build_full_name(n, idx):
    """Generate full operator name from operator list and site indices
    """
    
    res=""
    for k in range(len(idx)):
        res=res+n[k]+"_"+str(idx[k])
    return '{}'.format(res)


def get_opname(state):
    """Get operator name from Quspin basis state
    """
    name=""
    count=0
    indices=[]
    pos=0
    for op in state:
        tmpStr, tmpCount, tmpPos = opname(op)
        name = name + tmpStr
        count = count + tmpCount
        if(count>0 and tmpStr != ""):
            indices.append(pos)
        pos+=tmpPos
    indices = [i - indices[0] for i in indices]
    return name, count, indices, build_full_name(name, indices) 


def compare_idx(I,J):
    """Compare two indices
    """
    for k in range(len(I)):
        if I[k] != J[k]:
            return False
    return True


def samples_from_quspin(L,J,g,h,support=2, fileName=None, directoryName=None, numData=500, chargeChoice="1000"):
    """Generate observations from Gibbs matrices for transverse-field Ising model.

    Generated samples are directly saved to ``directoryName/fileName`` if those parameters are given.

    Arguments:
        * ``L``: System size
        * ``J``: Ising coupling
        * ``g``: Transverse field
        * ``support``: Support of observables
        * ``fileName``: Output file name
        * ``directoryName``: Output directory
        * ``numData``: Number of samples to generate
        * ``chargeChoice``: String indicating which charges to include

    Returns: Generated observables, list of observable names
    """

    charges=[s=="1" for s in chargeChoice]

    samples, op_names = generate_gge_ensemble_tIsing(L=L,K=J,G=g,charges=charges,support=support,numData=numData)
        
    samples = np.array(samples)

    if fileName is not None:
        if directoryName is not None:
            if not os.path.exists(directoryName):
                os.makedirs(directoryName)
            fileName = directoryName +"/"+ fileName

        # Save data
        with open(fileName,'w') as f:
            f.write("# Order of operators\n")
            for k, op in enumerate(op_names):
                f.write("# "+str(k)+" "+op+"\n")
            np.savetxt(f, samples)
        print("\nSaved data to " + fileName)

    result=(samples, op_names)

    return result


def generate_gge_ensemble_tIsing(L,K,G,charges=[True,False,False,False],support=2,numData=500,verbose=False): 
    """Generate observations from Gibbs matrices for quantum Ising model.

    Arguments:
        * ``L``: System size
        * ``K``: Ising coupling
        * ``G``: Transverse field
        * ``charges``: List of booleans indicating which charges to include
        * ``support``: Support of observables
        * ``numData``: Number of samples to generate
        * ``verbose``: Boolean indicating verbosity of Quspin

    Returns: Generated observables, list of observable names
    """

    # Construct basis
    basis=spin_basis_1d(L=L,a=1,kblock=0)

    dynamic=[]

    # Construct Operator basis
    opBasis=boson_basis_1d(support,sps=4)
    tmp_operator_list=[]
    op_names=[]
    for x in opBasis.states:
        tmpName, tmpCount, tmpIdx, tmpFullName = get_opname(opBasis.int_to_state(x))
        if tmpCount > 0:
            exists_already=False
            if len(tmp_operator_list) > 0:
                for k in range(len(tmp_operator_list)):
                    if tmpName==tmp_operator_list[k][0] and compare_idx(tmpIdx, tmp_operator_list[k][2]):
                        exists_already=True
            if not exists_already:
                op_names.append(tmpFullName)
                tmp_operator_list.append([tmpName, tmpCount, tmpIdx, tmpFullName])

    # Generate operators
    print("Generating operators:")
    operator_list=[]
    l=0
    for op in tmp_operator_list:
        conn=[]
        for i in range(L):
            tmpConn=[.5/L]
            for j in op[2]:
                tmpConn.append((i+j)%L) 
            conn.append(tmpConn)
        
        static=[[op[0],conn]]
        print(l, op[0], op[2])
        l=l+1
        operator_list.append(hamiltonian(static,dynamic,basis=basis,dtype=np.cdouble,check_herm=verbose,check_symm=verbose,static_fmt="csr"))
    
    # For progress bar
    sys.stdout.write("Generating data...\n")
    pbarWidth=40
    sys.stdout.write("[%s]" % (" " * pbarWidth))
    sys.stdout.flush()
    sys.stdout.write("\b" * (pbarWidth+1))

    data=[]
    for n in range(numData):
        # Draw random Lagrange multipliers in [-2:2]
        betas=4.*np.random.random((4))-2.

        # Hamiltonian
        term_e1_1=[[-K*betas[0],i,(i+1)%L] for i in range(L)] #PBC
        term_e1_2=[[-G*betas[0],i] for i in range(L)] 
        # first odd charge, C1
        term_o1_1 = [[-betas[1],i,(i+1)%L] for i in range(L)] 
        term_o1_2 = [[betas[1],i,(i+1)%L] for i in range(L)]          
        # second even charge C2
        term_e2_1 = [[K*betas[2],i,(i+1)%L,(i+2)%L] for i in range(L)] 
        term_e2_2 = [[-G*betas[2],i,(i+1)%L] for i in range(L)] 
        term_e2_3 = [[-G*betas[2],i,(i+1)%L] for i in range(L)] 
        term_e2_4 = [[-K*betas[2],i] for i in range(L)]
        # second odd charge, C3
        term_o2_1 = [[-betas[3],i,(i+1)%L,(i+2)%L] for i in range(L)] 
        term_o2_2 = [[betas[3],i,(i+1)%L,(i+2)%L] for i in range(L)]  

        # Assemble "Hamiltonian" = log(rho) operator
        static=[]
        if charges[0]:
            # Add Hamiltonian part=C_e1
            static.append(["zz",term_e1_1])
            static.append(["x",term_e1_2])
        if charges[1]:
            # Add C1
            static.append(["yz",term_o1_1])
            static.append(["zy",term_o1_2])        
        if charges[2]:
            # Add C2
            static.append(["zxz",term_e2_1])
            static.append(["yy",term_e2_2])
            static.append(["zz",term_e2_3])
            static.append(["x",term_e2_4])
        if charges[3]:
            # Add C3
            static.append(["yxz",term_o2_1])
            static.append(["zxy",term_o2_2]) 

        H=hamiltonian(static,dynamic,basis=basis,dtype=np.complex64,check_herm=verbose,check_symm=verbose)
        H_eigval, H_eigvec = H.eigh()

        # Get Gibbs density matrix        
        rho = H_eigvec.dot(np.diag(np.exp(H_eigval)).dot(np.transpose(np.conj(H_eigvec))))
        rho /= np.trace(rho)

        # Measure observables
        data.append([np.real(np.trace(op.static.dot(rho))) for op in operator_list])
        
        # For progress bar
        sys.stdout.write("#" * int(((n+1)*pbarWidth)/numData))
        sys.stdout.flush()
        sys.stdout.write("\b" * int(((n+1)*pbarWidth)/numData))

    return (data, op_names)


# Generate data
data=samples_from_quspin(L=L,J=J,h=h,g=g,support=support, fileName=fn, directoryName=dn, numData=numData, chargeChoice=chargeChoice)
