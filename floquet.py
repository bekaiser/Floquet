# 

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



from functions import make_Lap_inv, steady_nonrotating_solution, xforcing_nonrotating_solution, make_d, make_e, make_Lap_inv, make_partial_z, make_DI, make_D4, make_A13, make_A14, make_A34, make_A43, check_matrix, rk4, ordered_prod, time_step

figure_path = "/home/bryan/data/floquet/figures/"



# =============================================================================

# fluid properties
nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity

# flow characteristics
T = 44700.0 # s, M2 tide period
omg = 2.0*np.pi/T # rads/s
f = 1e-4 # 1/s, inertial frequency
N = 1e-3 # 1/s, buoyancy frequency
U = 0.001 # m/s, oscillation velocity amplitude
L = U/omg # m, excursion length
thtc= ma.asin(omg/N) # radians      #*180./np.pi # degrees
tht = 1./4.*thtc # radians
#tht = 1.0*np.pi/180. # rads, slope angle
Re = omg*L**2./nu

dRe = np.sqrt(2.*nu/omg)

# resolution
H = dRe*20.
#print(dRe,H)
Nz = 6 #int(H*10)
z = np.linspace((H/Nz)/2. , H, num=Nz) # m 
dz = z[1]-z[0] # m
#z = z + dz/2. # cell centers
print('dz = ', dz)

k0=1. # non-dimensional wavenumber
l0=1.


#print('Buoyancy CFL =', N*dt/dt)

# =============================================================================


Nt = int(T*1000) 

# time series
t = np.linspace( 0. , T*1. , num=Nt , endpoint=True , dtype=float) #[0.] 
dt = t[1]-t[0]

print('CFL =', U*dt/dz)
print('CFLx =', U*dt*np.sqrt(k0**2.+l0**2.))

Phi0 = np.eye(int(4*Nz),int(4*Nz),0,dtype=complex)
Tf = t[0]
#print(np.shape(Phin))


# time advancement:
Phin = time_step( Nz, N, omg, tht, nu, kap, U, t, z, dz, l0, k0, Phi0 , dt, Nt)

"""
for n in range(0,Nt):
  print(n)
  Tf=t[n]
  time = t[n]

  # Runge-Kutta, 4th order: 
  k1 = rk4( Nz, N, omg, tht, nu, kap, U, time , z, dz, l, k, Phin )
  k2 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l, k, Phin + k1*dt/2.)
  k3 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l, k, Phin + k1*dt/2.)
  k4 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt , z, dz, l, k, Phin + k3*dt)
  #k1 =  rk4( alph, beta, omg, time, Phin )
  #k2 =  rk4( alph, beta, omg, time + dt/2., Phin + k1*(dt/2.)  )
  #k3 =  rk4( alph, beta, omg, time + dt/2., Phin + k2*(dt/2.)  )
  #k4 =  rk4( alph, beta, omg, time + dt, Phin  + k3*dt )
  Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dt/6.; # now at t[n+1]
 
  print(time)

 
print(Tf)
"""

# computed solution:
w,v = np.linalg.eig(Phin) # eigenvals, eigenvecs | eigenvals = floquet multipliers
#print('computed mu1 = ', w[0])
#print('computed mu2 = ', w[1])

# checks w,v decomposition:
"""
print('Should be zero =',np.dot((Phin-np.eye(int(2),int(2),0,dtype=complex)*w[0]),v[:,0])) # C*v_k=lambda*I*v_k
print('Should be zero =',np.dot((Phin-np.eye(int(2),int(2),0,dtype=complex)*w[1]),v[:,1]))
"""

h5_filename = stat_path + 'eigvals.h5' #%(Ni[0,0],Ni[Nfiles-1,1])
f2 = h5py.File(h5_filename, "w")

# time series
dset = f2.create_dataset('Nt', data=Nt, dtype='f8')
dset = f2.create_dataset('Nz', data=Nz, dtype='f8')
dset = f2.create_dataset('k', data=k, dtype='f8')
dset = f2.create_dataset('l', data=l, dtype='f8')
dset = f2.create_dataset('w', data=w, dtype='f8')
dset = f2.create_dataset('v', data=v, dtype='f8')
dset = f2.create_dataset('U', data=U, dtype='f8')
dset = f2.create_dataset('N', data=N, dtype='f8')
dset = f2.create_dataset('tht', data=tht, dtype='f8')
dset = f2.create_dataset('Pr', data=Pr, dtype='f8')
dset = f2.create_dataset('omg', data=omg, dtype='f8')

#printname = '%i to %i' %(Ni[0,0],Ni[Nfiles-1,1])
print('\nfile written\n')


