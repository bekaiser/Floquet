#
# Bryan Kaiser


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


T = 2.*np.pi # radians, non-dimensional period
Td = 44700. # s, M2 tide period
Nz = 50 # number of grid points
grid_flag = 'cosine' # 'uniform' 
wall_flag = 'moving' 
#H = 1. # non-dimensional domain height (L is the lengthscale)
z,dz = fn.grid_choice( grid_flag , Nz , 1. ) # non-dimensional grid
# fixed variables
nu = 1.e-6 # m^2/s
omg = 2.*np.pi/44700. # rads/s
# variables to loop over:
#ReS = 1190 # 1192.8307471060361
k0 = 0.35+0.j # non-dimensional wavenumber
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
Hd = 5.*dS # m, dimensional domain height (arbitrary choice)
Ngrid = 1
#ReS = np.linspace(1200.,2400.,num=Ngrid,endpoint=True)
ReS = np.array([400.])
#tht = np.linspace(0.5,1.5,num=Ngrid,endpoint=True)*(2.*np.pi/180.)
CFL = 0.5

u_path = '/home/bryan/git_repos/Floquet/figures/base_flow/u/'
uz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/uz/'
uzz_path = '/home/bryan/git_repos/Floquet/figures/base_flow/uzz/'
phi_path = '/home/bryan/git_repos/Floquet/figures/phi/'

S = np.zeros([Ngrid,1]); # stability (1 = stable, 0 = unstable) ??
maxmod = np.zeros([Ngrid,1]); # maximum modulus of multipliers

count = 1

for i in range(0,1):
  for j in range(0,Ngrid):
    
    print(count)
    count = count + 1

    # dependent variables
    U = ReS[j] * np.sqrt(nu*omg/2.)
    L = U/omg

    print('non-dimensional period, T = ',T)
    print('dimensional period, Td = ',Td)
    print('dimensional excursion length, L = ',L)
    print('Stokes Reynolds number, ReS = ',ReS[j])
    print('dimensional domain height, Hd = ',Hd)
    print('oscillation amplitude, U = ',U)
    print('streamwise perturbation wavenumber, k = ',k0)
    print('Non-dimensional z_max = ',np.amax(z))
    print('Non-dimensional z_min = ',np.amin(z))

    
    dt1 = CFL * (U**.2/nu*omg) * np.amin(dz)**2.  #CFL = nu * omg * / (U^2) dt/dz**2 
    dt2 = CFL / (U*np.sqrt(2.*nu*omg)) * np.amin(dz)   # CFL = U*dt/dz = U*sqrt(2*nu*omg) * dt/dz
    #dt1 = CFL * ReS[j] * np.amin(dz)**2. # non-dimensional diffusion time step limit
    #dt2 = CFL * np.amin(dz) # non-dimensional advective time step limit
    dt = np.amin([dt1,dt2])
    Nt = int(T/dt)
    print('Number of time steps, Nt = ',Nt)

    params = {'nu': nu, 'omg': omg, 'L':L, 'T': T, 'Td': Td, 'Nt':Nt, 'U': U,  
          'Hd': Hd, 'Nz':Nz, 'wall':wall_flag,
          'dS':dS, 'Re':ReS[j], 'grid':grid_flag, 
          'u_path':u_path,'uz_path':uz_path,'uzz_path':uzz_path,
          'phi_path':phi_path,
          'z':z, 'dz':dz, 'k0':k0, 'CFL':CFL}

    Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)

    plotname = params['phi_path'] +'%i.png' %(1)
    fig = plt.figure(figsize=(16,4.5))
    plt.subplot(131); plt.plot(np.amax(abs(Phi0),0),params['z'],'b')
    plt.xlabel(r"$\Phi$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
    plt.ylim([-0.05,1.05]); plt.grid()
    plt.title(r"t/T = %.4f, step = %i" %(0./params['T'],00),fontsize=13)
    plt.subplot(132); plt.plot(np.amax(abs(Phi0),0),params['z'],'b')
    plt.xlabel(r"$\Phi$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
    plt.ylim([-0.001,0.03]); plt.grid()
    plt.title(r"t/T = %.4f, step = %i" %(0./params['T'],00),fontsize=13)
    plt.subplot(133); plt.semilogy(np.amax(abs(Phi0),0),params['z'],'b')
    plt.xlabel(r"$\Phi$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
    plt.ylim([0.,0.03]); plt.grid()
    plt.title(r"t/T = %.4f, step = %i" %(0./params['T'],00),fontsize=13)
    plt.savefig(plotname,format="png"); plt.close(fig);

    Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'blennerhassett' )


    mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
    maxmod[j,i] = np.amax(mod)
    print('maximum modulus = ',maxmod[j,i])
    for k in range(0,Nz):
      if round(mod[k],10) <= 1.:
        S[j,i] = 1. # 1 is for stability


print(maxmod)
print(S)

"""
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
"""
