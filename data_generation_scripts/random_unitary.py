#
# Script to generate training data from random unitary evolution
#
# Author: Markus Schmitt
# Date: Feb 2021
#
# Reference: 
#   Lanczos evolution with quspin https://weinbe58.github.io/QuSpin/examples/example20.html#example20-label
#

import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

import quspin
from quspin.basis import spin_basis_1d, boson_basis_1d
from quspin.operators import hamiltonian, quantum_LinearOperator
from scipy.sparse.linalg import expm_multiply
from quspin.tools.lanczos import lanczos_full as the_lanczos_fun
from quspin.tools.lanczos import expm_lanczos
import numpy as np

import matplotlib.pyplot as plt

import time

from functools import partial

# Prep. for Lanczos algorithm
lanczos_fun = partial(the_lanczos_fun, full_ortho=True)

#########
# Set simulation parameters here:

lanczosDim = 20             # Lanczos subspace dimension
dt = 1e-1                   # time step
nSteps = int(1./dt)         # number of steps needed to reach time 1

numSamples = 100            # Number of training data to generate

L = 20                      # system size
Delta = 1e-1                # Evolution time for random entangling gates

postSelectionCutoff=2e-2    # Cutoff on late time fluctuations for post-selection of samples

#########


# Read input from command line
if len(sys.argv)<2:
    raise Exception("Missing command line argument. Run with 'python random_unitary.py your/output/directory/ [optional seed]'")
    exit()

wdir = sys.argv[1]
if len(sys.argv)>2:
    np.random.seed(int(sys.argv[2]))


def get_operators(L,basis,support=3):
    """Generate all observables for given support.

    Arguments:
        * ``basis``: Basis of the spin chain (``quspin.basis.spin_basis_1d``)
        * ``support``: size of the subsystem (integer)

    Returns: List of operators, list of operator indices, list of operator names
    """

    o = ["x","y","z"]

    # Helper function to translate boson basis states to operators
    def get_opname(state):

        name=""
        count=0
        indices=[]
        pos=0
        for op in state:
            if op.isdigit():
                indices.append(int(op))

        return indices

    # Generate a boson basis, whose basis states correspond to
    # the operators that we need
    opBasis=boson_basis_1d(support,sps=4,pblock=1)

    # Translate quspin basis states to operator names
    ops=[]
    for x in opBasis.states:
        tmp = get_opname(opBasis.int_to_state(x))
        if tmp[0] != 0:
            ops.append(tmp)

    # Generate all operators
    operators=[]
    opIdx = {}
    opNames = []
    for r,op in enumerate(ops):

        n = ""
        idx = []
        name = ""
        for l,a in enumerate(op):
            if int(a)>0:
                n += o[int(a)-1]
                idx.append(l)
                name += o[int(a)-1]+str(l)

        for k in range(2-idx[-1]):
            tmpIdx = [i+k+1 for i in idx]
            tmpName = ""
            for j,i in enumerate(tmpIdx):
                tmpName += name[2*j]+str(i)
            opIdx[tmpName] = r

        opIdx[name] = r
        opNames.append(name)
        print(n, idx)
        I = [[0.5/L, *[(i+l)%L for l in idx]] for i in range(L)]
        opDescr = [[n,I],[n[::-1],I]]
        operators.append(quantum_LinearOperator(opDescr, basis=basis, dtype=np.complex128))

    return operators, opIdx, opNames


def get_connected(obs, opIdx, opNames):
    """Translates full list of correlation functions to connected correlation functions

    Arguments:
        * ``obs``: observations (list of list of float)
        * ``opIdx``: Operator indices (as produced by ``get_operators``)
        * ``opNames``: Operator names (as produced by ``get_operators``)

    Returns: Connected version of the input observations
    """

    res = []
    for i,o in enumerate(obs):
        n = opNames[i]
        if len(n) == 2:
            res.append(o)
            continue
        if len(n) == 4:
            res.append(o - obs[opIdx[n[:2]]]*obs[opIdx[n[2:]]])
            continue
        if len(n) == 6:
            tmp = o + 2*obs[opIdx[n[:2]]]*obs[opIdx[n[2:4]]]*obs[opIdx[n[4:]]]
            tmp -= obs[opIdx[n[:2]]]*obs[opIdx[n[2:]]]
            tmp -= obs[opIdx[n[:4]]]*obs[opIdx[n[4:]]]
            tmp -= obs[opIdx[n[:2]+n[4:]]]*obs[opIdx[n[2:4]]]
            res.append(tmp)

    return res


def get_random_rotation(gateType, rndRange, L, basis):
    """Generate Hamiltonian for a random rotation

    Arguments:
        * ``gateType``: Rotation axis ("x", "y", or "z")
        * ``rndRange``: Interval of possible angles (list [min,max])
        * ``L``: System size (integer)
        * ``basis``: Basis of the spin chain (``quspin.basis.spin_basis_1d``)

    Returns: Hamiltonian that generates the rotation when evolving from t=0 to t=1.
    """

    C = np.random.uniform(rndRange[0],rndRange[1],1)[0]

    cpl = [[C,l] for l in range(L)]
    static = [[gateType, cpl]]

    return hamiltonian(static,[],basis=basis,dtype=np.float64, check_symm=False, check_herm=False)


def get_random_entangler(rndRange, L, basis, evenOdd):
    """Generate Hamiltonian for a random two-qubit gate

    Arguments:
        * ``rndRange``: Interval of possible angles (list [min,max])
        * ``L``: System size (integer)
        * ``basis``: Basis of the spin chain (``quspin.basis.spin_basis_1d``)
        * ``evenOdd``: Gates to be applied on even or odd bonds? (integer)

    Returns: Hamiltonian that generates the rotation when evolving from t=0 to t=1.
    """

    evenOdd = evenOdd % 2

    thetas = np.random.uniform(rndRange[0],rndRange[1],3)

    C1 = 0.5 * (thetas[0] + thetas[1])
    C2 = - 0.5 * (thetas[0] - thetas[1])
    C3 = 0.5 * thetas[2]

    cpl1 = [[C1,2*l+evenOdd,(2*l+evenOdd+1)%L] for l in range(L//2)]
    cpl2 = [[C2,2*l+evenOdd,(2*l+evenOdd+1)%L] for l in range(L//2)]
    cpl3 = [[C3,l] for l in range(L)]
    static = [["+-", cpl2],["-+", cpl2], ["zz", cpl1], ["z", cpl3]]

    return hamiltonian(static,[],basis=basis,dtype=np.float64, check_symm=False, check_herm=False)


def random_rotation(psi, L, basis):
    """Perform random rotation

    Arguments:
        * ``psi``: Wave function (numpy array)
        * ``L``: System size (integer)
        * ``basis``: Basis of the spin chain (``quspin.basis.spin_basis_1d``)

    Returns: Transformed wave function
    """

    X = get_random_rotation("x", [0,2*np.pi], L, basis)
    Z = get_random_rotation("z", [0,np.pi], L, basis)

    for i in range(nSteps):    
    
        E,V,Q = lanczos_fun(X,psi,lanczosDim)
        psi = expm_lanczos(E,V,Q,a=-1.j*dt)
        
    for i in range(nSteps):    

        E,V,Q = lanczos_fun(Z,psi,lanczosDim)
        psi = expm_lanczos(E,V,Q,a=-1.j*dt)

    return psi

def random_entangler(psi, L, basis, evenOdd):
    """Apply random two-qubit gate

    Arguments:
        * ``psi``: Wave function (numpy array)
        * ``L``: System size (integer)
        * ``basis``: Basis of the spin chain (``quspin.basis.spin_basis_1d``)
        * ``evenOdd``: Gates to be applied on even or odd bonds? (integer)

    Returns: Transformed wave function
    """

    ZZ = get_random_entangler([-np.pi,np.pi], L, basis, evenOdd)

    E,V,Q = lanczos_fun(ZZ,psi,lanczosDim)
    psi = expm_lanczos(E,V,Q,a=-1.j*Delta) # !!! Attention: for larger Delta more steps might be required !!!

    return psi


def measure(psi,op):
    """Get expectation value of operator.

    Arguments:
        * ``psi``: Wave function (numpy array)
        * ``op``: Operator (quspin.operators.quantum_LinearOperator)

    """
    return np.real(op.expt_value(psi))


# Initialize basis
basis = spin_basis_1d(L,a=2,kblock=0,pblock=1,pauli=True)
print("\nHilbert space dimension: {}.\n".format(basis.Ns))

# Generate operators
opList, opIdx, opNames = get_operators(L,basis,support=3)

with open(wdir+"operators.txt", 'w') as f:
    f.write("List of operators:\n")
    for k, opn in enumerate(opNames):
        f.write(str(k)+" "+opn+"\n")

# Generate a set of "probe" operators used to monitor evolution
s = [[1./L, i] for i in range(L)]
nn = [[1./L, i,(i+1)%L] for i in range(L)]
X = hamiltonian([["x",s]], [], basis=basis, dtype=np.float64)
Y = hamiltonian([["y",s]], [], basis=basis, dtype=np.complex128)
XX = hamiltonian([["zz",nn]], [], basis=basis, dtype=np.float64)
Z = hamiltonian([["z",s]], [], basis=basis, dtype=np.float64)

data=[]

t=0
mdt=0.2 # Measurement interval for "probe" operators
nMeas=0

# Set time points at which training data is measured (time unit is 0.1/step)
dataPoints=np.array([0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.2, 5.0, 10.0, 20.0, 30.0, 50.0])
T=dataPoints[-1]+2*dt

dataCounter = 0

stepId=0
for sampleId in range(numSamples):

    print("** %d" % (sampleId), flush=True)
    
    # Initialize wave function
    psi = np.zeros(basis.Ns)
    psi[0] = 1.
    psi = random_rotation(psi, L, basis)
    psi = random_rotation(psi, L, basis)

    # Measure Z for later check of U(1) symmetry
    M0 = np.real(np.conj(psi).dot(Z.dot(psi)))

    testData = []
    observations = []
    states = []

    print("Magnetization density: %.3f" % (M0))
    t = 0
    nMeas = 0
    tic = time.perf_counter()
    while t<T:    

        # measure local magnetizations towards end of evolution to later check fluctuations
        if t>0.85*dataPoints[-1] and (t-0.85*dataPoints[-1])>nMeas*mdt:
            Xval = measure(psi,X)
            Yval = measure(psi,Y)
            testData.append([t, Xval, Yval])
            nMeas+=1

        # Store states at sampling time points
        if np.any(np.abs(t - dataPoints)<1e-4):
            states.append(psi.copy())

        # Perform time step
        psi = random_entangler(psi, L, basis, stepId)
        stepId+=1
        psi = random_entangler(psi, L, basis, stepId)
        stepId+=1
        
        t+=dt

    # Measure Z in final wave function
    M1 = np.real(np.conj(psi).dot(Z.dot(psi)))

    print("Magnetization difference: %.6f" % (np.abs(M1-M0)))
    print("Time evolution took %fs" % (time.perf_counter()-tic))

    # compute standard deviation as measure for fluctuations at late times
    fluct=np.std(np.array(testData)[:,1:], axis=0)

    print("Late time fluctuations: %f %f" % (fluct[0], fluct[1]))

    # Post-selection of samples: Discard too strongly fluctuating samples
    if not np.any(fluct>postSelectionCutoff):
        tic = time.perf_counter()
        # compute and store all observables
        for r,p in zip(dataPoints, states):
            obs = np.array([get_connected([measure(p,op) for op in opList],opIdx,opNames)])
            obs[np.where(np.abs(obs)<1e-13)] = 0.
            with open(wdir+"obs_t=%.1f.txt"%(r), 'a') as f:
                np.savetxt(f, obs)
        dataCounter += 1
        print("  Measurements took %fs" % (time.perf_counter()-tic))
        print("  # post-selected samples: %d" % (dataCounter))

