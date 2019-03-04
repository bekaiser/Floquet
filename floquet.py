# Floquet analysis of a stratified Boussinesq flow
# Bryan Kaiser
# 

# non-dimensionalize the U component!!!

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import expm
#from scipy.stats import chi2
#from scipy import signal
#from scipy.fftpack import fft, fftshift
#import matplotlib.patches as mpatches
#from matplotlib.colors import colorConverter as cc
import functions as fn
from datetime import datetime 
import numpy.distutils.system_info as sysinfo
sysinfo.get_info('atlas')


figure_path = "./figures/"
stat_path = "./"


# =============================================================================
# parameters 

k0=0. # streamwise non-dimensional perturbation wavenumbers
l0=0. # spanwise non-dimensional perturbation wavenumbers

Nz = 200 # number of grid points
grid_flag = 'cosine' # 'uniform'

nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
T = 44700.0 # s, M2 tide period
omg = 2.0*np.pi/T # rads/s
#f = 1e-4 # 1/s, inertial frequency
N = 1e-3 # 1/s, buoyancy frequency
C = 1./4. # N^2*sin(tht)/omg, slope ``criticality''
U = 0.01 # m/s, oscillation velocity amplitude
L = U/omg # m, excursion length (here we've assumed L>>H)
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians
Re = omg*L**2./nu # Reynolds number
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
ReS = np.sqrt(2.*Re) # Stokes' 2nd problem Reynolds number
H = L # non-dimensional domain height (L is the lengthscale)
z,dz = fn.grid_choice( grid_flag , Nz , H )


#print(np.amax(z),np.amin(z))
print('ReS = ', ReS)
print('C = ',tht/thtc)
print('Pr = ',Pr)
print('k = ', k0)
print('l = ', l0)

# time series:
CFL = 0.02
dt = np.amin(dz)*CFL # non-dimensional time step

print('dimensional dt = ',dt*T)
print('approx number of time steps = ',1./dt)

params = {'nu': nu, 'kap': kap, 'Pr': Pr, 'omg': omg, 'L':L, 'T': T, 'U': U, 
          'N':N, 'tht':tht, 'Re':Re, 'C':C, 'H':H, 'Nz':Nz, 'k0':k0, 'l0':l0,
          'dS':dS, 'ReS':ReS, 'thtc':thtc, 'grid':grid_flag, 
          'dz_min':(np.amin(dz)),'dt':dt, 'CFL':(dt/np.amin(dz))}


# time advancement:
Phi0 = np.eye(int(4*Nz),int(4*Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
start_time = datetime.now()
Phin = fn.rk4_time_step( params, z, Phi0 , dt, 1. )
time_elapsed = datetime.now() - start_time
print('Total time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# Floquet mode/multiplier solutions:
eigval,eigvec = np.linalg.eig(Phin) # eigenvals, eigenvecs | eigenvals = floquet multipliers

eigvalr = np.real(eigval)
eigvali = np.imag(eigval)
eigvecr = np.real(eigvec)
eigveci = np.imag(eigvec)
#print(eigval)
#print(eigvec)

#print(eigvalr)
#print(eigvali)

#print(type(Pr))
#print(type(C))
Prandtl = Pr

# save results to .h5:
h5_filename = stat_path + 'eigvals.h5' 
f2 = h5py.File(h5_filename, "w")
dset = f2.create_dataset('ReS', data=ReS, dtype='f8')
dset = f2.create_dataset('Prandtl', data=Prandtl, dtype='f8')
dset = f2.create_dataset('C', data=C, dtype='f8')
dset = f2.create_dataset('dt', data=dt, dtype='f8')
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

#print(eigvalr)
#print(eigvali)
#print(np.shape(eigvalr))

modulus = np.power( np.power(eigvalr,2.) + np.power(eigvali,2.) , 0.5 )
eigvalr = np.sort(abs(eigvalr))
eigvali = np.sort(abs(eigvali))
modulus = np.sort(modulus)
 
Neig = np.shape(eigvalr)[0]
print(modulus[Neig-1])
#print(eigvalr)
print(eigvalr[Neig-1]) 
print(eigvali[Neig-1])
Nch = 200

plotname = figure_path +'modulus.png' 
fig = plt.figure(figsize=(8,5))
plt.plot(np.arange(Nch,0,-1),modulus[Neig-Nch:Neig],'b')
#plt.hist(y_test_pred_f, color = 'red', edgecolor = 'black',bins = binsize2, density = True, alpha = 0.2,label=r"prediction") # [:,0]
#plt.hist(test_log10eps, color = 'blue', edgecolor = 'black', bins = binsize2, density = True, alpha = 0.25,label=r"data")
plt.xlabel(r"$N$",fontsize=13)
plt.ylabel(r"|$\lambda$|",fontsize=13)
#plt.title(r"test data, $N_{profiles}=$%i" %(Nfiles2),fontsize=13)
#plt.legend(loc=1)
#plt.xlim([-12.5,-6.5]) 
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path +'real_eigenvalues.png' 
fig = plt.figure(figsize=(8,5))
plt.plot(np.arange(Nch,0,-1),eigvalr[Neig-Nch:Neig],'b')
#plt.hist(y_test_pred_f, color = 'red', edgecolor = 'black',bins = binsize2, density = True, alpha = 0.2,label=r"prediction") # [:,0]
#plt.hist(test_log10eps, color = 'blue', edgecolor = 'black', bins = binsize2, density = True, alpha = 0.25,label=r"data")
plt.xlabel(r"$N$",fontsize=13)
plt.ylabel(r"|$\lambda$|",fontsize=13)
#plt.title(r"test data, $N_{profiles}=$%i" %(Nfiles2),fontsize=13)
#plt.legend(loc=1)
#plt.xlim([-12.5,-6.5]) 
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'imag_eigenvalues.png' 
fig = plt.figure(figsize=(8,5))
plt.plot(np.arange(Nch,0,-1),eigvali[Neig-Nch:Neig],'b')
#plt.hist(y_test_pred_f, color = 'red', edgecolor = 'black',bins = binsize2, density = True, alpha = 0.2,label=r"prediction") # [:,0]
#plt.hist(test_log10eps, color = 'blue', edgecolor = 'black', bins = binsize2, density = True, alpha = 0.25,label=r"data")
plt.xlabel(r"$N$",fontsize=13)
plt.ylabel(r"|$\lambda$|",fontsize=13)
#plt.title(r"test data, $N_{profiles}=$%i" %(Nfiles2),fontsize=13)
#plt.legend(loc=1)
#plt.xlim([-12.5,-6.5]) 
plt.savefig(plotname,format="png"); plt.close(fig);



