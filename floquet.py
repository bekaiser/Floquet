# Floquet analysis of a stratified Boussinesq flow
# Bryan Kaiser
# 

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


# non-dimensional perturbation wavenumbers, non-dimensionalized by L=U/omega:
k0=2.*np.pi 
l0=2.*np.pi

Nz = 100 # number of grid points
H = 20. # non-dimensional domain height
grid = 'cosine'
#grid = 'uniform'

nu = 2.0e-2 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
T = 44700.0 # s, M2 tide period
omg = 2.0*np.pi/T # rads/s
#f = 1e-4 # 1/s, inertial frequency
N = 1e-3 # 1/s, buoyancy frequency
C = 1./4.
U = 0.01 #0.00592797 # m/s, oscillation velocity amplitude
"""
 U = [0.00592797, 0.01185594, 0.02371189, 0.05927972, 0.11855945] corresponds to 
 ReS = U*np.sqrt(2./(nu*omg)) =
 [500.,1000.,2000.,5000.,10000.]
"""

L = U/omg # m, excursion length
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians, sets C = 1/4
Re = omg*L**2./nu # Reynolds number
dRe = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
ReS = np.sqrt(2.*Re)
print(dRe/H)

if grid == 'uniform': 
 z = np.linspace((H/Nz)/2. , H, num=Nz) # non-dimensional

if grid == 'cosine': 
 z = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H
 z = z[0:Nz] # half cosine grid
 dz = z[1:Nz]-z[0:Nz-1]

print(np.amax(z),np.amin(z))
print('ReS = ', ReS)
print('C = ',tht/thtc)
print('Pr = ',Pr)
print('H =', H)
print('k = ', k0)
print('l = ', l0)

params = {'nu': nu, 'kap': kap, 'Pr': Pr, 'omg': omg, 'L':L, 'T': T, 'U': U, 'N':N, 'tht':tht, 'Re':Re, 'C':C, 'H':H, 'Nz':Nz, 'k0':k0, 'l0':l0}

# time series:
Nt = 10000 
t = np.linspace( 0. , T*1. , num=Nt , endpoint=True , dtype=float)/T #[0.] 
dt = t[1]-t[0]
print(dt,np.amin(dz))
print('CFL =', dt/np.amin(dz))
#print('CFLx =', U*dt*np.sqrt(k0**2.+l0**2.))

# time advancement:
Phi0 = np.eye(int(4*Nz),int(4*Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
start_time = datetime.now()
#Phin = op_time_step( Nz, N, omg, tht, nu, kap, U, z, l0, k0, Phi0 , dt, 100. ) 
Phin = fn.rk4_time_step( params, z, Phi0 , dt, 1. )
time_elapsed = datetime.now() - start_time
print('Total time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# Floquet mode/multiplier solutions:
eigval,eigvec = np.linalg.eig(Phin) # eigenvals, eigenvecs | eigenvals = floquet multipliers
"""
# checks w,v decomposition:
print('Should be zero =',np.dot((Phin-np.eye(int(2),int(2),0,dtype=complex)*w[0]),v[:,0])) # C*v_k=lambda*I*v_k
"""
eigvalr = np.real(eigval)
eigvali = np.imag(eigval)
eigvecr = np.real(eigvec)
eigveci = np.imag(eigvec)
#print(eigval)
#print(eigvec)

#print(eigvalr)
#print(eigvali)

print(type(Pr))
print(type(C))
Prandtl = Pr

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
print(np.shape(eigvalr))


