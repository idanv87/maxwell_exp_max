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

from scipy.sparse.linalg import cg, spsolve
import pylops
import timeit

import ray

from scipy import signal
import matplotlib.pyplot as plt

class Calc:
       sigma=0.1
       def __init__(self,p, U,D,f,Rho,Psi):
        self.U = U
        self.D=D
        self.f=f
        self.Rho=Rho
        self.Psi=Psi
        self.p=p

       
       def base(self,m,s):
           return np.exp(-(np.linalg.norm(self.U[m]-self.p@self.D[s]@self.f)**2)/self.sigma)
          
       
       def Adm(self,m):
           return sum([self.base(m,s)*self.Rho[s] for s in range(len(self.D))])
       
       
       def A(self,m,s):
           return self.base(m,s)*self.Rho[s]/self.Adm(m)
       
       def create_v(self,i):
           x=0


           for m in range(len(self.U)):
               for s in range(len(self.D)):
                    x+=self.inner(self.U[m],self.p@self.D[s]@self.Psi[i])
           return x        
       
       @staticmethod
       def inner(u,v):
           return np.dot(np.transpose(v),u)
                   
       def create_A(self,i,j):
           x=0
           for m in range(len(self.U)):
               for s in range(len(self.D)):
                   x+=self.inner(self.p@self.D[s]@self.Psi[i],self.p@self.D[s]@self.Psi[j])
           return x              
       
       def solve(self):
           A=np.zeros((len(self.Psi),len(self.Psi)))
           V=np.zeros((len(self.Psi),1))
           for i in range(len(self.Psi)):
               
               V[i]=self.create_v(i)
               for j in range(len(self.Psi)):
                   A[i,j]=self.create_A(i,j)
                   
           return scipy.linalg.solve(A,V)
       
       def up_rho(self):
           rho_new=np.zeros((len(D),1))
           for s in range(len(D)):
               x=0
               for m in range(len(self.U)):
                   x+=(1/(len(self.D)))*self.A(m,s)
               
               rho_new[s]=x

           return rho_new
                   
           
N1=10
N2=100     
psi=[np.random.rand(N2,1), np.random.rand(N2,1) ]
alpha=[0.1,0.2]
u=  [np.random.rand(N1,1)]   
D=[np.random.rand(N2,N2)]
rho=[1]
p=np.random.rand(N1,N2)
f=np.random.rand(N2,1)

def exp_max(p, u,D,alpha,rho,psi,num_iter=4):
    assert len(rho)==len(D)
    assert len(alpha)==len(psi)

    for _ in range(num_iter):
        f=sum([alpha[i]*psi[i] for i in range(len(psi))])
        C=Calc(p, u,D,f,rho,psi)
        alpha=C.solve()
        rho=C.up_rho()
        print(rho)
  


    return alpha, rho

# a,b=exp_max(p, u,D,alpha,rho,psi)
# print(a)
# print(b)





def dft_mtx(n):
    return scipy.linalg.dft(n)/np.sqrt(n)

def create_D2(n):
   
    kernel = np.zeros((n, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)

    D2=csr_matrix(D2)
    return D2

N=30
N_samples=5
def create_A(dt,h,n):
    D=create_D2(n).toarray()/h**2
    A=np.block([[np.zeros((n,n)),np.eye(n)],[D,np.zeros((n,n))]])
    return np.eye(2*n)+dt*A
p=np.zeros((N_samples,2*N))
p[:N_samples, :N_samples]=np.eye(N_samples)


x=np.linspace(0,1,N+1)[1:]
dx=x[1]-x[0]
dt=np.sqrt(0.1)*dx
f=np.sin(2*math.pi*x).reshape(N,1)
g=f*0
f0=np.vstack([f,g])

mtx=create_A(dt,dx,N)
psi=[np.vstack([f,g]),0.5*np.vstack([np.sin(4*math.pi*x).reshape(N,1),g])] # base functions 
alpha=[0.1,0.01]
rho=[0.4,0.4,0.2]
u=[]
D=[mtx,mtx@mtx, mtx@mtx@mtx]
for i in range(10):
    f0=mtx@f0
    if i>7:
        u.append(p@f0)


a,b=exp_max(p, u,D,alpha,rho,psi)
print(a)
