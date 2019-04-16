
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

Nz = 600 # number of grid points
grid_flag = 'cosine' # 'uniform'
wall_flag = 'moving' 
#wall_flag = 'farfield' 

nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
#kap = nu/Pr # m^2/s, thermometric diffusivity
omg = 2.0*np.pi/Td # rads/s
#f = 0. #1e-4 # 1/s, inertial frequency
#N = 1e-3 # 1/s, buoyancy frequency
#C = 1./4. # N^2*sin(tht)/omg, slope ``criticality''
U = 0.01 # m/s, oscillation velocity amplitude
L = U/omg # m, excursion length (here we've assumed L>>H)
Re = omg*L**2./nu # Reynolds number
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
ReS = np.sqrt(2.*Re) # Stokes' 2nd problem Reynolds number
H = 1.
Hd = 100.*dS # m, dimensional domain height 
z,dz = fn.grid_choice( grid_flag , Nz , 1. )
wall_flag = 'moving'

Nt = 100 # number of time steps
dt = T/(Nt-1) # non-dimensional dt

params = {'nu': nu, 'omg': omg, 'L':L, 'T': T, 'Td': Td, 'U': U, 'H': H, 'Hd': Hd,
          'Re':Re, 'H':H, 'Nz':Nz, 'wall_flag':wall_flag,
          'dS':dS, 'ReS':ReS,
          'dz_min':(np.amin(dz)),'Nt':Nt, 'z':z, 'dz':dz}


# non-rotating solutions
uA = np.zeros([Nz,Nt]);  
uzA = np.zeros([Nz,Nt]);  
uzzA = np.zeros([Nz,Nt]);  

uzC = np.zeros([Nz,Nt]); 
uzzC = np.zeros([Nz,Nt]);


t = np.zeros([Nt])
time = 0.
for n in range(0,Nt):
  #print(time)

  t[n] = time # non-dimensional time, time = [0,2pi]
  u,uz,uzz = fn.stokes_solution( params, time, 2 ) # dimensional 
  uA[:,n] = u / params['U']
  uzA[:,n] = uz / (np.sqrt(2.)*params['U']) * params['dS'] 
  uzzA[:,n] = uzz / (2.*params['U']) * params['dS']**2.
  
  dz,lBC = fn.partial_z( params , 'dirchlet' , 'neumann' ) # dimensional 
  uzC[:,n] = np.dot( np.real( dz ) , u ) # dimensional 
  uzC[0,n] = uzC[0,n] + lBC * 2.* np.real(params['U']*np.exp(1j*time))

  dzz,lBC2 = fn.partial_zz( params , 'dirchlet' , 'neumann' ) # dimensional
  #dzz,lBC2 = fn.partial_zz_old( z*Hd, Nz , Hd , 'dirchlet' , 'neumann' ) # dimensional
  uzzC[:,n] = np.dot( np.real( dzz ) , u ) # dimensional 
  #uzzC[:,n] = np.dot( dz , np.dot( np.real( dz ) , u ) )# dimensional 
  uzzC[0,n] = uzzC[0,n] + lBC2 * 2. * params['U'] * np.cos(time) #np.real(params['U']*np.exp(1j*time))  

  time = time + dt # non-dimensional time

print(lBC,lBC2)

# dimensional solutions:
uzA_dim = uzA * (np.sqrt(2.)*params['U']) / params['dS'] 
uzzA_dim = uzzA * (2.*params['U']) / params['dS']**2.
uzC_dim = uzC
uzzC_dim = uzzC

# non-dimensional solutions:
uzC = uzC / (np.sqrt(2.)*params['U']) * params['dS']
uzzC = uzzC / (2.*params['U']) * params['dS']**2.

zmaxzoom = 0.2

A,B = np.meshgrid(t/T,z)



plotname = figure_path +'uz_dim.png'
fig = plt.figure(figsize=(16,4.5))
plt.subplot(131); 
CS = plt.contourf(A,B,uzA_dim,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(132); 
CS = plt.contourf(A,B,uzC_dim,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(133); 
CS = plt.contourf(A,B,abs(uzA_dim-uzC_dim),200,cmap='Greys')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);



plotname = figure_path +'uz.png'
fig = plt.figure(figsize=(16,4.5))
plt.subplot(131); 
CS = plt.contourf(A,B,uzA,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(132); 
CS = plt.contourf(A,B,uzC,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(133); 
CS = plt.contourf(A,B,abs(uzA-uzC),200,cmap='Greys')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path +'uzz.png'
fig = plt.figure(figsize=(16,4.5))
plt.subplot(131); 
CS = plt.contourf(A,B,uzzA,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(132); 
CS = plt.contourf(A,B,uzzC,200,cmap='seismic')
plt.colorbar(CS)
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(133); 
CS = plt.contourf(A,B,abs(uzzA-uzzC),200,cmap='Greys')
plt.colorbar(CS)
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);



plotname = figure_path +'uzz_dim.png'
fig = plt.figure(figsize=(16,4.5))
plt.subplot(131); 
CS = plt.contourf(A,B,uzzA_dim,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"$z/H$",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(132); 
CS = plt.contourf(A,B,uzzC_dim,200,cmap='seismic')
plt.colorbar(CS)
plt.axis([0.,1.,0.,zmaxzoom])
plt.subplot(133); 
CS = plt.contourf(A,B,abs(uzzA_dim-uzzC_dim),200,cmap='Greys')
plt.colorbar(CS)
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);





#Binf = U*(N**2.0)*np.sin(tht)/omg  


### u

"""
plotname = figure_path + wall_flag + '_u_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"analytical non-rotating u/U" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,u/U,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,u/U,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);
"""

