#
# Bryan Kaiser
# 3/21/2019

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

# check types of matrices in A
# check that the BL is resolved
# check turning off diffusion
# verify max/min of u in base flow

# =============================================================================

# the resulting eigensystem is not a function of Re,Pr or T.

T = 2.*np.pi # radians, non-dimensional period
Td = 44700. # s, M2 tide period
Nz = 20 # number of grid points
grid_flag = 'cosine' # 'uniform' 
wall_flag = 'moving' 
Pr = 1. # Prandtl number
H = 1. # non-dimensional domain height (L is the lengthscale)
z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid


# fixed variables
nu = 1.e-6 # m^2/s
omg = 2.*np.pi/44700. # rads/s
N = 1e-3 # 1/s 



# variables to loop over:
#ReS = 1190 # 1192.8307471060361
#C = 0.25                       
Pr = 1.
Ro = np.inf   # np.inf 1.406502996287728
k0 = 1. # non-dimensional wavenumber
l0 = 0. # non-dimensional wavenumber

Ngrid = 10
ReS = np.linspace(5.,5000.,num=Ngrid,endpoint=True)
C = np.linspace(0.05,0.5,num=Ngrid,endpoint=False)

S = np.zeros([Ngrid,Ngrid]);
maxmod = np.zeros([Ngrid,Ngrid]);

count = 1

for i in range(0,Ngrid):
  for j in range(0,Ngrid):
 
    print(count)
    count = count + 1

    # dependent variables
    tht = ma.asin(C[i]*omg/N)
    kap = nu/Pr
    Re = ReS[j]**2./2.
    dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
    f = omg / (np.cos(tht)*Ro)
    U = ReS[j] * np.sqrt(nu*omg/2.)
    L = U/omg
    Hd = 100.*dS # m, dimensional domain height (arbitrary choice)
    

    print('non-dimensional period, T = ',T)
    print('dimensional period, Td = ',Td)
    print('Reynolds number, Re = ',Re)
    print('Stokes Reynolds number, ReS = ',ReS[j])
    print('criticality, C =',C[i])
    print('Rossby number, Ro =',Ro)
    print('Prandtl number, Pr = ',Pr)
    print('non-dimensional domain height, H = ',H)
    print('dimensional domain height, Hd = ',Hd)
    print('oscillation amplitude, U = ',U)
    print('streamwise perturbation wavenumber, k = ',k0)
    print('spanwise perturbation wavenumber, l = ',l0)
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


    mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
    maxmod[j,i] = np.amax(mod)
    print('maximum modulus = ',maxmod[j,i])
    for k in range(0,Nz):
      if round(mod[k],10) <= 1.:
        S[j,i] = 1. # 1 is for stability


print(S)
A,B = np.meshgrid(ReS,C) 

plotname = figure_path +'stability_advdiff.png' 
plottitle = r"advection diffusion stablity, k=%.2f,l=%.2f" %(k0,l0) 
fig = plt.figure(figsize=(8,8))
CS = plt.contourf(A,B,S,cmap='gist_gray')
plt.colorbar(CS)
plt.xlabel(r"Re$_S$",fontsize=13);
plt.ylabel(r"C",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'maxmod_advdiff.png' 
plottitle = r"advection diffusion max modulus, k=%.2f,l=%.2f" %(k0,l0) 
fig = plt.figure(figsize=(8,8))
CS = plt.contourf(A,B,maxmod,cmap='gist_gray')
plt.colorbar(CS)
plt.xlabel(r"Re$_S$",fontsize=13);
plt.ylabel(r"C",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);
