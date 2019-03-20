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

T = 2.*np.pi #44700. # s, M2 tide period

Nz = 200 # number of grid points
grid_flag = 'cosine' # 'uniform'
wall_flag = 'moving' 
#wall_flag = 'farfield' 

nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
omg = 2.0*np.pi/T # rads/s
f = 0. #1e-4 # 1/s, inertial frequency
N = omg/0.14056342969081848 # 1/s, buoyancy frequency
C = 1./4. # N^2*sin(tht)/omg, slope ``criticality''
U = 0.01 # m/s, oscillation velocity amplitude
L = U/omg # m, excursion length (here we've assumed L>>H)
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians
Re = omg*L**2./nu # Reynolds number
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
ReS = np.sqrt(2.*Re) # Stokes' 2nd problem Reynolds number
H = 30.*dS # m, dimensional domain height (L is the lengthscale)
z,dz = fn.grid_choice( grid_flag , Nz , H )


#CFL = 0.02
#dt = np.amin(dz)*CFL # non-dimensional time step
#Nt = 1000 #T/dt
#dt = T/(Nt-1) 'dz_min':(np.amin(dz)), 'CFL':(dt/np.amin(dz)),

k0 = 1.
l0 = 1.

params = {'nu': nu, 'kap': kap, 'omg': omg, 'L':L, 'T': T, 'U': U, 'H': H, 
          'N':N, 'tht':tht, 'Re':Re, 'C':C, 'H':H, 'Nz':Nz, 'wall':wall_flag,
          'dS':dS, 'ReS':ReS, 'thtc':thtc, 'grid':grid_flag, 'f': f, 'Pr':Pr,
          'z':z, 'dz':dz,'k0': k0, 'l0': l0, 'wall':wall_flag}






Ngrid = 1

strutt = np.zeros([Ngrid,Ngrid]); #strutt2 = np.zeros([Ngrid,Ngrid])

count = 1

for i in range(0,Ngrid):
  for j in range(0,Ngrid):
 
    print(count)
    count = count + 1

    Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
    Phin,final_time = fn.rk4_time_step( params, Phi0 , T/5000, T , 'buoyancy_equation' )


    mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
    if mod[0] < 1. and mod[1] < 1.:
      strutt[j,i] = 1. # stable
    print(mod)
    """
    modOPH = np.abs(np.linalg.eigvals(PhinOPH)) # eigenvals = floquet multipliers
    if modOPH[0] < 1. and modOPH[1] < 1.:
      strutt4[j,i] = 1.
    """

print(strutt)

"""
A,B = np.meshgrid(np.log10(k),np.log10(c))

plotname = figure_path +'strutt_buoy_eig_rk4.png' 
plottitle = r"inviscid buoyancy equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt,cmap='gist_gray')
plt.xlabel(r"log$_{10}(k)$ non-dimensional wavenumber",fontsize=13);
plt.ylabel(r"log$_{10}$C",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);
"""




"""
plotname = figure_path +'strutt_eig_op.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt4,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);
"""
