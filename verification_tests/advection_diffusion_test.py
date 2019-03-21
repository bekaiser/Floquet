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

T = 2.*np.pi # radians, non-dimensional period
Td = 44700. # s, M2 tide period
Nz = 20 # number of grid points
grid_flag = 'uniform' # 'cosine' 
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
k0 = 0.1 # non-dimensional wavenumber
l0 = 0. # non-dimensional wavenumber

# dependent variables
tht = ma.asin(C*omg/N)
kap = nu/Pr
Re = ReS**2./2.
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
f = omg / (np.cos(tht)*Ro)
U = ReS * np.sqrt(nu*omg/2.)
L = U/omg
Hd = 100.*dS # m, dimensional domain height (arbitrary choice)

print('Non-dimensional period, T = ',T)
print('Dimensional period, Td = ',Td)
print('Reynolds number, Re = ',Re)
print('Stokes Reynolds number, ReS = ',ReS)
print('Rossby number, Ro =',Ro)
print('Prandtl number, Pr = ',Pr)
print('Non-dimensional domain height, H = ',H)
print('Dimensional domain height, Hd = ',Hd)
print('Oscillation amplitude, U = ',U)
print('Non-dimensional z_max = ',np.amax(z))
print('Non-dimensional z_min = ',np.amin(z))

CFL = 0.25 
dt1 = CFL * Pr * Re * np.amin(dz)**2. # non-dimensional diffusion time step limit
dt2 = CFL * np.amin(dz) # non-dimensional advective time step limit
dt = np.amin([dt1,dt2])
Nt = int(T/dt)
print('Number of time steps, Nt = ',Nt)

params = {'nu': nu, 'kap': kap, 'omg': omg, 'L':L, 'T': T, 'Td': Td, 'Nt':Nt, 'U': U, 
          'N':N, 'tht':tht, 'Re':Re, 'C':C, 'H': H, 'Hd': Hd, 'Nz':Nz, 'wall':wall_flag,
          'dS':dS, 'ReS':ReS, 'grid':grid_flag, 'f': f, 'Pr':Pr,
          'z':z, 'dz':dz, 'Ro':Ro, 'k0':k0, 'l0':l0, 'CFL':CFL}

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
print('unstable if less than zero:',np.sum(S)-Nz)


