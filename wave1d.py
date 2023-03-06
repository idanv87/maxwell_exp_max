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

from utils import create_p, generator

class Calc:
       sigma=0.1
       def __init__(self,p, U,D,f,Rho,Psi, iter):
        self.iter=iter
        self.U = U
        self.D=D
        self.f=f
        self.Rho=Rho
        self.Psi=Psi
        self.p=p

       
       def base(self,m,s):
           return np.exp(-(np.linalg.norm(self.U[m]-self.p@np.linalg.matrix_power(self.D[s],self.iter)@self.f)**2)/self.sigma)
          
       
       def Adm(self,m):
           return sum([self.base(m,s)*self.Rho[s] for s in range(len(self.D))])
       
       
       def A(self,m,s):
           return self.base(m,s)*self.Rho[s]/self.Adm(m)
       
       def create_v(self,i):
           x=0


           for m in range(len(self.U)):
               for s in range(len(self.D)):
                    x+=self.inner(self.U[m],self.p@np.linalg.matrix_power(self.D[s],self.iter)@self.Psi[i])
           return x        
       
       @staticmethod
       def inner(u,v):
           return np.dot(np.transpose(v),u)
       
                   
       def create_A(self,i,j):
           x=0
           for m in range(len(self.U)):
               for s in range(len(self.D)):
                   x+=self.inner(self.p@np.linalg.matrix_power(self.D[s],self.iter)@self.Psi[i],self.p@np.linalg.matrix_power(self.D[s],self.iter)@self.Psi[j])
           return x              
       
       def solve(self):
           A=np.zeros((len(self.Psi),len(self.Psi)))
           V=np.zeros((len(self.Psi),1))
           for i in range(len(self.Psi)):
               
               V[i]=self.create_v(i)
               for j in range(len(self.Psi)):
                   A[i,j]=self.create_A(i,j)
           print(np.linalg.norm(A,2))
           return scipy.linalg.pinv(A)@V  
        #    return scipy.linalg.solve(A,V)
       
       def up_rho(self):
           rho_new=np.zeros((len(D),1))
           for s in range(len(D)):
               x=0
               for m in range(len(self.U)):
                   x+=(1/(len(self.U)))*self.A(m,s)
               
               rho_new[s]=x

           return rho_new
                   
           
# N1=10
# N2=100     
# psi=[np.random.rand(N2,1), np.random.rand(N2,1) ]
# alpha=[0.1,0.2]
# u=  [np.random.rand(N1,1)]   
# D=[np.random.rand(N2,N2)]
# rho=[1]
# p=np.random.rand(N1,N2)
# f=np.random.rand(N2,1)

def exp_max(p, u,D,alpha,rho,psi,iter,num_iter=1):
    assert len(rho)==len(D)
    assert len(alpha)==len(psi)
    
    for _ in range(num_iter):
        alpha_old=alpha
        f=sum([alpha[i]*psi[i] for i in range(len(psi))])
        C=Calc(p, u,D,f,rho,psi, iter)
        alpha=C.solve()
        rho=C.up_rho()
        # print(np.linalg.norm(alpha-alpha_old))
       
  


    return alpha, rho

# a,b=exp_max(p, u,D,alpha,rho,psi)
# print(a)
# print(b)





def dft_mtx(n):
    return scipy.linalg.dft(n)/np.sqrt(n)

def create_bc(n):
    kernel = np.zeros((n, 1))

    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1

    return circulant(kernel)


def create_D2(n,w):
    
    kernel = np.zeros((n, 1))
    kernel[-2]=w[2]
    kernel[-1] = w[1]
    kernel[0] = w[0]
    kernel[1] = w[1]
    kernel[2]=w[2]
    D2 = circulant(kernel)
    D2[:2,:]=create_bc(n)[:2,:]
    D2[-2:,:]=create_bc(n)[-2:,:]
    D2=csr_matrix(D2)
    
    return D2




def create_A(dt,h,n,w):

    D=create_D2(n,w).toarray()/h**2
    A=np.block([[np.zeros((n,n)),np.eye(n)],[D,np.zeros((n,n))]])
    # return np.linalg.inv(np.eye(2*n)-dt*A)
    return np.eye(2*n)+dt*A


N=32
N_samples=2
iter=10

G=generator(N,N_samples,0.01,int(N/2))
p=G.p
psi=G.psi[::2]
alpha=[1/len(psi) for _ in range(len(psi))]
dx=G.dx
dt=G.dt

rho=[1]
u=[]
w1=[-2,1,0]
w2=[-30/12, 16/12,-1/12]
mtx=create_A(dt,dx,N,w1)
D=[mtx]
f0=psi[4]
for i in range(50):
    f0=mtx@f0
    if i>iter:
        u.append(p@f0+np.random.normal(0, 0.5, size=(N_samples, 1)))


alpha_final,rho_final=exp_max(p, u,D,alpha,rho,psi, iter+2)
# print(sum(rho_final))
# # print(rho_final)
print(np.argmax(abs(alpha_final)))
print(abs(alpha_final))
plt.plot(psi[np.argmax(alpha_final)])
plt.plot(psi[4])
plt.show()
# x=np.linspace(0,1,N+1)[1:]
# dx=x[1]-x[0]
# f=np.sin(2*math.pi*x).reshape(N,1)
# D=create_D2(N,w1).toarray()/dx/dx
# print(mean_squared_error(D@f,-4*math.pi**2*f))
# err=[]
# for i in range(int(1/dt)):
#     err.append(mean_squared_error(f0[3:N-3],(f_an*np.cos(2*math.pi*(i)*dt))[3:N-3]))
#     f0=mtx@f0
   

#     if True:
#         u.append(p@f0)

# print(np.mean(err))
# plt.show()