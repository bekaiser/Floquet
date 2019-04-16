#
# Bryan Kaiser
# 3/21/2019

# change old base flow test to test of discrete derivatives, with modified pz,pzz functions
# fix moving wall v


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
Nz = 200 # number of grid points
grid_flag = 'cosine' # 'uniform' 
wall_flag = 'moving' 
Pr = 1. # Prandtl number
H = 1. # non-dimensional domain height (L is the lengthscale)
z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid


# fixed variables
nu = 1.e-6 # m^2/s
omg = 2.*np.pi/44700. # rads/s
N = 1e-3 # 1/s 
f = 1e-4 
kap = nu/Pr

thtc = ma.asin(omg/N)
tht = 0.25*thtc
Ro = omg / ( f * np.cos(tht) )

Ri = N**2./omg**2. # Richardson number

# variables to loop over:
#ReS = 1190 # 1192.8307471060361
#C = 0.25                       
Pr = 1.   # np.inf 1.406502996287728
k0 = 1. # non-dimensional wavenumber
l0 = 0. # non-dimensional wavenumber

dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
#f = omg / (np.cos(tht)*Ro)

u_path = '/home/bryan/git_repos/Floquet/figures/base_flow/u/'
uz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/uz/'
uzz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/uzz/'
v_path = '/home/bryan/git_repos/Floquet/figures/base_flow/v/'
vz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/vz/'
vzz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/vzz/'
b_path = '/home/bryan/git_repos/Floquet/figures/base_flow/b/'
bz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/bz/'
bzz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/bzz/'

Hd = 100.*dS # m, dimensional domain height (arbitrary choice)

Ngrid = 1
ReS = np.linspace(50.,150.,num=Ngrid,endpoint=True)
tht = np.linspace(0.5,1.5,num=Ngrid,endpoint=True)*(2.*np.pi/180.)

S = np.zeros([Ngrid,Ngrid]); # stability (1 = stable, 0 = unstable) ??
maxmod = np.zeros([Ngrid,Ngrid]); # maximum modulus of multipliers

#tht = 2.*np.pi/180.*1. # 1 degree slope
#Pr = 1.

count = 1

for i in range(0,Ngrid):
  for j in range(0,Ngrid):
 
    print(count)
    count = count + 1

    # dependent variables
    #tht = ma.asin(C[i]*omg/N)
    
    #print(np.shape(kap),np.shape(omg))
    Re = ReS[j]**2./2.
    U = ReS[j] * np.sqrt(nu*omg/2.)
    L = U/omg
    Ro = omg / ( f*np.cos(tht[i]) )

    print('slope angle, theta =',tht[i])
    print('non-dimensional period, T = ',T)
    print('dimensional period, Td = ',Td)
    print('dimensional excursion length, L = ',L)
    print('Reynolds number, Re = ',Re)
    print('Stokes Reynolds number, ReS = ',ReS[j])
    #print('criticality, C =',C[i])
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
    #dt = np.amin([dt1,dt2])
    Nt = 10 #int(T/dt)
    dt = T/Nt
    print('Number of time steps, Nt = ',Nt)

    params = {'nu': nu, 'kap': kap, 'omg': omg, 'L':L, 'T': T, 'Td': Td, 'Nt':Nt, 'U': U, 
          'N':N, 'tht':tht[i], 'Re':Re, 'Ri':Ri, 'Ro':Ro, 'dt':dt,
          'u_path':u_path,'uz_path':uz_path,'uzz_path':uzz_path,
          'v_path':v_path,'vz_path':vz_path,'vzz_path':vzz_path,
          'b_path':b_path,'bz_path':bz_path,'bzz_path':bzz_path,   
          'H': H, 'Hd': Hd, 'Nz':Nz, 'wall':wall_flag,
          'dS':dS, 'ReS':ReS, 'grid':grid_flag, 'f': f, 'Pr':Pr,
          'z':z, 'dz':dz, 'Ro':Ro, 'k0':k0, 'l0':l0, 'CFL':CFL}

    Phi0 = np.eye(1,1,0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
    Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'base_flow_test' )

    """
    mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
    maxmod[j,i] = np.amax(mod)
    print('maximum modulus = ',maxmod[j,i])
    for k in range(0,Nz):
      if round(mod[k],10) <= 1.:
        S[j,i] = 1. # 1 is for stability
    """

#print(S)
#A,B = np.meshgrid(ReS,C) 

"""
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
"""
