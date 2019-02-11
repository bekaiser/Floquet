# Floquet analysis of an oscillating boundary layer flow
# Bryan Kaiser
# 2/5/19

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chi2
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from functions import make_Lap_inv, steady_nonrotating_solution, xforcing_nonrotating_solution, make_d, make_e, make_Lap_inv, make_partial_z, make_D, make_A13, make_A14, make_A34, make_A43, check_matrix, rk4, ordered_prod, time_step, adaptive_time_step
from datetime import datetime 
import numpy.distutils.system_info as sysinfo
sysinfo.get_info('atlas')


figure_path = "./figures/"
stat_path = "./"


# =============================================================================


# fluid properties:
nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
print('Pr = ',Pr)

# flow characteristics:
T = 44700.0 # s, M2 tide period
omg = 2.0*np.pi/T # rads/s
#f = 1e-4 # 1/s, inertial frequency
N = 1e-3 # 1/s, buoyancy frequency
U = 0.00592797 # m/s, oscillation velocity amplitude
# U = [0.00592797, 0.01185594, 0.02371189, 0.05927972, 0.11855945] corresponds to 
# ReS = U*np.sqrt(2./(nu*omg)) =
# [500.,1000.,2000.,5000.,10000.]
L = U/omg # m, excursion length
thtc= ma.asin(omg/N) # radians    
C = 1./4.
tht = C*thtc # radians, sets C = 1/4
Re = omg*L**2./nu # Reynolds number
dRe = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
ReS = np.sqrt(2.*Re)
print('ReS = ', ReS)
print('C = ',tht/thtc)
#
Ngrid = np.array([2000,1000,500,200,100,50,20,10]) #10,20,50,100,200,500,1000,2000]) #np.fliplr(np.array([10,20,50,100,200,500,1000,2000]))
mean_eigvalr = np.zeros([8])
mean_eigvali = np.zeros([8])

for q in range(0,8):

 # grid:
 H = dRe*40.
 Nz = int(Ngrid[q]) #int(H*10)
 print(Nz)
 z = np.linspace((H/Nz)/2. , H, num=Nz) # m 
 dz = z[1]-z[0] # m
 print('dz = ', dz)
 print(H)

 # non-dimensional perturbation wavenumbers, non-dimensionalized by L=U/omega:
 k0=2.*np.pi 
 l0=2.*np.pi
 # [2pi,128pi,256pi,1024pi,2048pi]
 print('k = ', k0)
 print('l = ', l0)


 # time series:
 Nt = int(T*100) 
 t = np.linspace( 0. , T*1. , num=Nt , endpoint=True , dtype=float) #[0.] 
 dt = t[1]-t[0]
 print('CFL =', U*dt/dz)
 print('CFLx =', U*dt*np.sqrt(k0**2.+l0**2.))


 # time advancement:
 Phi0 = np.eye(int(4*Nz),int(4*Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
 start_time = datetime.now()
 Phin = time_step( Nz, N, omg, tht, nu, kap, U, z, dz, l0, k0, Phi0 , dt, 1. )
 time_elapsed = datetime.now() - start_time
 print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

 # Floquet mode/multiplier solutions:
 eigval,eigvec = np.linalg.eig(Phin) # eigenvals, eigenvecs | eigenvals = floquet multipliers
 """
 # checks w,v decomposition:
 print('Should be zero =',np.dot((Phin-np.eye(int(2),int(2),0,dtype=complex)*w[0]),v[:,0])) # C*v_k=lambda*I*v_k
 print('Should be zero =',np.dot((Phin-np.eye(int(2),int(2),0,dtype=complex)*w[1]),v[:,1]))
 """
 eigvalr = np.real(eigval)
 eigvali = np.imag(eigval)
 #eigvecr = np.real(eigvec)
 #eigveci = np.imag(eigvec)
 #print(eigval)
 #print(eigvec)

 print(Nz)
 print(np.shape(eigvalr))
 mean_eigvalr[q] = np.mean(eigvalr,0) 
 mean_eigvali[q] = np.mean(eigvali,0) 

 print(type(Pr))
 print(type(C))
 Prandtl = Pr

mean_eigvalr = mean_eigvalr/mean_eigvalr[0]
mean_eigvali = mean_eigvali/mean_eigvali[0]

print(np.shape(mean_eigvalr))
print(np.shape(Ngrid))

plotname = figure_path +'local_loglog_error.png' 
plottitle = r"resolution" 
fig = plt.figure()
plt.loglog(Ngrid,mean_eigvalr,'b',label="real")
plt.loglog(Ngrid,mean_eigvali,'r',label=r"imag")
plt.xlabel(r"N",fontsize=13);
plt.ylabel(r"error",fontsize=13); 
plt.legend(loc=2,fontsize=14); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'local_semilogy_error.png' 
plottitle = r"resolution" 
fig = plt.figure()
plt.semilogy(Ngrid,mean_eigvalr,'b',label="real")
plt.semilogy(Ngrid,mean_eigvali,'r',label=r"imag")
plt.xlabel(r"N",fontsize=13);
plt.ylabel(r"error",fontsize=13); 
plt.legend(loc=2,fontsize=14); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'local_plot_error.png' 
plottitle = r"resolution" 
fig = plt.figure()
plt.plot(Ngrid,mean_eigvalr,'b',label="real")
plt.plot(Ngrid,mean_eigvali,'r',label=r"imag")
plt.xlabel(r"N",fontsize=13);
plt.ylabel(r"error",fontsize=13); 
plt.legend(loc=2,fontsize=14); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);


"""

# save results to .h5:
h5_filename = stat_path + 'eigvals.h5' 
f2 = h5py.File(h5_filename, "w")
dset = f2.create_dataset('ReS', data=ReS, dtype='f8')
dset = f2.create_dataset('Prandtl', data=Prandtl, dtype='f8')
dset = f2.create_dataset('C', data=C, dtype='f8')
dset = f2.create_dataset('t', data=t, dtype='f8')
dset = f2.create_dataset('z', data=z, dtype='f8')
dset = f2.create_dataset('k', data=k0, dtype='f8')
dset = f2.create_dataset('l', data=l0, dtype='f8')
dset = f2.create_dataset('eigvalr', data=eigvalr, dtype='f8')
dset = f2.create_dataset('eigvali', data=eigvali, dtype='f8')
dset = f2.create_dataset('eigvecr', data=eigvecr, dtype='f8')
dset = f2.create_dataset('eigveci', data=eigveci, dtype='f8')
dset = f2.create_dataset('U', data=U, dtype='f8')
dset = f2.create_dataset('N', data=N, dtype='f8')
dset = f2.create_dataset('tht', data=tht, dtype='f8')
dset = f2.create_dataset('Pr', data=Pr, dtype='f8')
dset = f2.create_dataset('omg', data=omg, dtype='f8')
print('\nfile written!\n')

print(eigvalr)
print(eigvali)

"""
