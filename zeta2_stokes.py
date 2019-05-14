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

# need a resolution requirement. From the analytical solution?

T = 2.*np.pi # s, period
omg = 2.*np.pi/T # rads/s
nu = 1e-6
Re = 700.
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
U = Re * (nu/dS) # Re = U*dS/nu, so ReB=Re/2
H = 300. # = Hd/dS, non-dimensional grid height
Hd = H*dS # m, dimensional domain height (arbitrary choice)
Nz = 200 # number of grid points
z,dz = fn.grid_choice( 'cosine' , Nz , H ) # non-dimensional grid
#print('non-dimensional grid min/max: ',np.amin(z),np.amax(z))
grid_flag = 'cosine' # 'uniform' 
wall_flag = 'moving' 

a = 0.38 # disturbance wavenumber
params0 = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U,  
          'Nz':Nz, 'Re':Re,'a':a, 'H':H, 'Hd':Hd,
          'dS':dS, 'z':z, 'dz':dz} #'grid_scale':grid_scale}
dzz_zeta = fn.diff_matrix( params0 , 'neumann' , 'dirchlet' , diff_order=2 , stencil_size=3 ) # non-dimensional
# dzz_zeta: could try neumann LBC. Upper BC irrotational (no-stress).
inv_psi = np.linalg.inv( fn.diff_matrix( params0 , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 ) ) # non-dimensional
# inv_psi: lower BCs are no-slip, impermiable, upper BC is impermiable, free-slip
eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex )

params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U, 'inv_psi':inv_psi,  
          'Nz':Nz, 'Re':Re,'a':a, 'H':H, 'Hd':Hd, 'dzz_zeta':dzz_zeta,
          'dS':dS, 'z':z, 'dz':dz, 'eye_matrix':eye_matrix} 

CFL = 0.5
#dt1 = CFL*(np.amin(dz*Hd))*omg # s*rads/s = rads, non-dimensional dt
#dt2 = CFL*((np.amin(dz*Hd))**2./nu)*omg # s*rads/s = rads, non-dimensional dt
dt1 = CFL*((dz[2]*Hd))*omg # s*rads/s = rads, non-dimensional dt
dt2 = CFL*(((dz[2]*Hd))**2./nu)*omg # s*rads/s = rads, non-dimensional dt
dt = np.amin([dt1,dt2])
if dt == dt1:
  print('Advective CFL')
if dt == dt2:
  print('Diffusive CFL')
Nt = int(T/dt)
#print(Nt)
#Nt = 2000 
print('Number of time steps, Nt = ',Nt)



Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'blennerhassett' )
mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
maxmod = np.amax(mod)
#print('maximum modulus = ',maxmod)
S = 0.
if maxmod <= 1.:
  S = 1. # 1 is for stability


print('Maximum modulus: ',maxmod)
if S == 1.:
  print('Floquet stable')
else:
  print('Floquet unstable')

print('Number of time steps, Nt = ',Nt)




     




