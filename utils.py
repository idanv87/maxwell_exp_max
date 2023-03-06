import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import multiprocessing
import timeit
import datetime
from scipy import sparse
from sympy import *
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import scipy 
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import circulant
from scipy.sparse import block_diag
from scipy.sparse import vstack
from sklearn.metrics import mean_squared_error

from scipy.sparse.linalg import cg, spsolve
import pylops
import timeit

import ray

from scipy import signal
import matplotlib.pyplot as plt

class generator:
    def __init__(self,N,N_samples,cfl, x0):
        self.x0=x0
        self.N=N
        self.x=np.linspace(0,1,self.N+1)[1:]
        self.dx=self.x[1]-self.x[0]
        self.dt=cfl*self.dx
        self.psi=self.base_functions()
        self.p=create_p(N_samples, N)
        self.dt=cfl*self.dx
        self.source=self.c_source()
        

    
    def c_source(self):
        f=np.zeros((self.N,1))
        f[int(self.x0)]=1
        g=f*0
        return np.vstack([f,g]) 
    
    
    def base_functions(self):
        
        psi=[]
        for i in range(3,self.N-3):
            f=np.zeros((self.N,1))
            f[i]=1
            f[i-1]=1
            f[i+1]=1
            psi.append(np.vstack([f,f*0]) )
        return psi    
    
    def create(u):
        pass 
    
    



def create_p(N_samples,N):
    assert N_samples==2
    x=np.zeros((N_samples,N))
    x[:N_samples, int(N/3)]=np.eye(N_samples)[:,0]
    x[:N_samples, int(2*N/3)]=np.eye(N_samples)[:,1]
    # np.random.shuffle(np.transpose(x))
    p=np.hstack((x,np.zeros((N_samples,N))))
    return p

# print(create_p(2,13)[:,:13])


