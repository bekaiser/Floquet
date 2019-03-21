# Stability calculation test
# Bryan Kaiser
# 3/14/2019

# Note: see LaTeX document "floquet_primer" for analytical solution derivation

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy   
from scipy import signal
import functions as fn

figure_path = "./figures/"


# =============================================================================

# the resulting eigensystem is not a function of Re,Pr or T.

T = 1. # non-dimensional period
Nz = 20 # number of grid points
grid_flag = 'uniform' # 'cosine' # 'uniform'#
wall_flag = 'moving' 
Pr = 1. # Prandtl number
H = 1. # non-dimensional domain height (L is the lengthscale)
Re = 50. # Reynolds number
z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid


# fixed variables
nu = 1.e-6 # m^2/s
omg = 2.*np.pi/44700. # rads/s
N = 1e-3 # 1/s 

# variables to loop over:
ReS = 1190 # 1192.8307471060361
C = 0.25                #         
Pr = 1.
Ro = np.inf   # np.inf 1.406502996287728

# dependent variables
tht = ma.asin(C*omg/N)
kap = nu/Pr
Re = ReS**2./2.
f = omg / (np.cos(tht)*Ro)
U = ReS * np.sqrt(nu*omg/2.)
L = U/omg

params = {'H': H, 'grid':grid_flag, 'wall':wall_flag, 'z':z, 'Nz':Nz, 'dz':dz, 'wall':wall_flag, 
          'Re':Re, 'Pr':Pr, 'kap':kap, 'nu':nu, 'omg':omg, 'N':N, 'U':U, 'L':L, 'f':f, 'tht':tht }

# z and t need to be dimensional when feed to rotating solution

# C = N*sin(tht)/omg
# Re 



print('Period = ',T)
print('Reynolds number = ',Re)
print('Prandtl number = ',Pr)
print('Domain height = ',H)
print('z_max = ',np.amax(z))
print('z_min = ',np.amin(z))

CFL = 0.25 
dt1 = CFL * Pr * Re * np.amin(dz)**2. # non-dimensional diffusion time step limit
dt2 = CFL * np.amin(dz) # non-dimensional advective time step limit
dt = np.amin([dt1,dt2])
#print(dt1,dt2,dt)
Nt = int(T/dt)
#print('Nt = ',Nt)

"""

Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'advection_diffusion' )

S = np.zeros([Nz])
mod,vec = np.linalg.eig(Phin)
mod = np.abs(mod)
for j in range(0,Nz):
  if round(mod[j],10) <= 1.:
    S[j] = 1. 


print(mod)
print(np.shape(mod))
print(np.shape(vec))
print(vec[:,0])
print(S)
print(round(mod[0],10))
print('unstable if less than zero:',np.sum(S)-Nz)

"""
