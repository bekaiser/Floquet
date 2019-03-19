
# Bryan Kaiser
# 3/  /2019


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

T = 44700. # s, M2 tide period

Nz = 200 # number of grid points
grid_flag = 'cosine' # 'uniform'
wall_flag = 'moving' 
#wall_flag = 'farfield' 

nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
omg = 2.0*np.pi/T # rads/s
f = 1e-4 # 1/s, inertial frequency
N = 1e-3 # 1/s, buoyancy frequency
C = 1./4. # N^2*sin(tht)/omg, slope ``criticality''
U = 0.01 # m/s, oscillation velocity amplitude
L = U/omg # m, excursion length (here we've assumed L>>H)
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians
Re = omg*L**2./nu # Reynolds number
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
ReS = np.sqrt(2.*Re) # Stokes' 2nd problem Reynolds number
H = 100.*dS # m, dimensional domain height (L is the lengthscale)
z,dz = fn.grid_choice( grid_flag , Nz , H )

#print(np.amin(z),np.amax(z))

#CFL = 0.02
#dt = np.amin(dz)*CFL # non-dimensional time step
Nt = 100 #T/dt
dt = T/(Nt-1)

#print('dimensional dt = ',dt*T)
#print('approx number of time steps = ',1./dt)

params = {'nu': nu, 'kap': kap, 'omg': omg, 'L':L, 'T': T, 'U': U, 'H': H,
          'N':N, 'tht':tht, 'Re':Re, 'C':C, 'H':H, 'Nz':Nz, 'wall':wall_flag,
          'dS':dS, 'ReS':ReS, 'thtc':thtc, 'grid':grid_flag, 'f': f, 'Pr':Pr,
          'dz_min':(np.amin(dz)),'Nt':Nt, 'CFL':(dt/np.amin(dz)), 'z':z, 'dz':dz}


u = np.zeros([Nz,Nt]); b = np.zeros([Nz,Nt]); 
uz = np.zeros([Nz,Nt]); bz = np.zeros([Nz,Nt]); 
t = np.zeros([Nt])

ur = np.zeros([Nz,Nt]); vr = np.zeros([Nz,Nt]); br = np.zeros([Nz,Nt]); 

uz_check = np.zeros([Nz,Nt]);
bz_check = np.zeros([Nz,Nt]);


time = 0.

for n in range(0,Nt):

  t[n] = time
  b[:,n],u[:,n],bz[:,n],uz[:,n] = fn.nonrotating_solution( params, time )
  br[:,n],ur[:,n],vr[:,n] = fn.rotating_solution( params, time )
  time = time + dt

  if wall_flag == 'moving':
    uz_check[:,n] = np.dot( np.real(fn.partial_z( params, 'neumann', 'neumann' )) , u[:,n] ) 
    # lower bc doesn't work for the moving wall, but not necessary
  if wall_flag == 'farfield':
    uz_check[:,n] = np.dot( np.real(fn.partial_z( params, 'dirchlet', 'neumann' )) , u[:,n] )
  bz_check[:,n] = np.dot( np.real(fn.partial_z( params, 'neumann' , 'neumann' )) , b[:,n] )

zmaxzoom = 0.002

A,B = np.meshgrid(t/T,z/H)

plotname = figure_path + wall_flag + '_u_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"non-rotating u/U" 
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

plotname = figure_path + wall_flag + '_ur_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"rotating u/U" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,ur/U,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,ur/U,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + wall_flag + '_vr_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"rotating v/U" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,vr/U,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,vr/U,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);


Binf = U*(N**2.0)*np.sin(tht)/omg  #L*N**2.*np.sin(tht)
#print(np.amin(b/Binf),np.amax(b/Binf))

plotname = figure_path + wall_flag + '_b_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"non-rotating b/($LN^2\sin\theta$)" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,b/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,b/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + wall_flag + '_br_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"rotating b/($LN^2\sin\theta$)" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,br/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,br/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + wall_flag + '_uz_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"$u_z\delta$/U" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,uz*dS/U,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,uz*dS/U,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + wall_flag + '_uz_solution_check.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"$u_z\delta$/U" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,uz_check*dS/U,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,uz_check*dS/U,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + wall_flag + '_bz_solution.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"$b_z\delta$/($LN^2\sin\theta$)" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,bz*dS/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,bz*dS/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path +  wall_flag + '_bz_solution_check.png' 
fig = plt.figure(figsize=(12,5))
plottitle = r"$b_z\delta$/($LN^2\sin\theta$)" 
plt.subplot(2, 1, 1)
CS = plt.contourf(A,B,bz_check*dS/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,1.])
plt.title(plottitle);
plt.subplot(2, 1, 2)
CS = plt.contourf(A,B,bz_check*dS/Binf,200,cmap='seismic')
plt.colorbar(CS)
plt.xlabel(r"t/T",fontsize=13);
plt.ylabel(r"z/H",fontsize=13); 
plt.axis([0.,1.,0.,zmaxzoom])
plt.savefig(plotname,format="png"); plt.close(fig);

