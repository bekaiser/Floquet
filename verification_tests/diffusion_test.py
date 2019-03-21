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

# eigensystem is not a function of Re or T.

T = 1. # s, period
Nz = 20 # number of grid points
grid_flag = 'cosine' # 'uniform'#
wall_flag = 'moving' 
Pr = 1. # Prandtl number
H = 1. # m, dimensional domain height (L is the lengthscale)
Re = 50. #Re = omg*H**2./nu # Reynolds number
z,dz = fn.grid_choice( grid_flag , Nz , H )
params = {'H': H,'grid':grid_flag,'z':z, 'Nz':Nz, 'dz':dz,'wall':wall_flag, 'Re':Re, 'Pr':Pr}

print('Reynolds number = ',Re)
print('Prandtl number = ',Pr)
print('Domain height = ',H)
print('z_max = ',np.amax(z))
print('z_min = ',np.amin(z))

CFL = 0.25
#CFL = 1/RePr * dt/dz^2 (non-dim dt and dz)
dt = np.amin(dz)**2. * (H**2.*Re*Pr*CFL) # non-dimensional time step
Nt = int(1./dt)
print('Nt = ',Nt)

Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'buoyancy_diffusion' )

S = np.zeros([Nz])
mod,vec = np.linalg.eig(Phin)
mod = np.abs(mod)
for j in range(0,Nz):
  if round(mod[j],15) <= 1.:
    S[j] = 1. 

print(mod)
print(np.shape(mod))
print(np.shape(vec))
print(vec[:,0])
print(S)
print(round(mod[0],15))
print('unstable if less than zero:',np.sum(S)-Nz)

