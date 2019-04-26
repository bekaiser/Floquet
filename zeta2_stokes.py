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
H = 200. # = Hd/dS, non-dimensional grid height
Hd = H*dS # m, dimensional domain height (arbitrary choice)
Nz = 200 # number of grid points
z,dz = fn.grid_choice( 'cosine' , Nz , H ) # non-dimensional grid
print('non-dimensional grid min/max: ',np.amin(z),np.amax(z))
grid_flag = 'cosine' # 'uniform' 
wall_flag = 'moving' 
#print(Hd/dS)
#grid_scale = Hd/dS

a = 0.38 # disturbance wavenumber
params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U,  
          'Nz':Nz, 'Re':Re,'a':a, 'H':H, 'Hd':Hd,
          'dS':dS, 'z':z, 'dz':dz} #'grid_scale':grid_scale}


CFL = 0.5
dt1 = CFL*(np.amin(dz*Hd))*omg # s*rads/s = rads, non-dimensional dt
dt2 = CFL*((np.amin(dz*Hd))**2./nu)*omg # s*rads/s = rads, non-dimensional dt
dt = np.amin([dt1,dt2])
#print(dt1,dt2)
Nt = int(T/dt)
Nt = 2000 
print('Number of time steps, Nt = ',Nt)


Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'blennerhassett' )
mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
maxmod = np.amax(mod)
#print('maximum modulus = ',maxmod)
S = 0.
if maxmod <= 1.:
  S = 1. # 1 is for stability

# CHOOSE NEUMANN BOTTOM, 

# neumann bottom: Nt =1000, H=100, Nz=100 
# Floquet stable for a=0.38, Re=500
# Floquet stable for a=0.38, Re=1000, 0.9995823422111871
# Floquet stable for a=0.38, Re=2000
# Floquet unstable for a=0.38, Re=5000

# neumann bottom: Nt =1000, H=125, Nz=100 
# Floquet stable for a=0.38, Re=800, 0.9988554793753912
# Floquet stable for a=0.38, Re=1000, 0.9990844184760211

# neumann bottom: Nt =1000, H=125, Nz=120
# Floquet stable for a=0.38, Re=1000, 0.9991110887269946

# neumann bottom: Nt =1000, H=125, Nz=200
# Floquet stable for a=0.38, Re=400, 0.999529423009477
# Floquet stable for a=0.38, Re=450, 0.9998000489556929
# Floquet stable for a=0.38, Re=490, 0.9999688651909235
# Floquet stable for a=0.38, Re=498, 0.9999985082966509
# Floquet unstable for a=0.38, Re=499, 1.000002126255642
# Floquet unstable for a=0.38, Re=500, 1.000005725159982
# Floquet unstable for a=0.38, Re=600, 1.0002852093030317
# Floquet unstable for a=0.38, Re=700, 1.000449765229297
# Floquet unstable for a=0.38, Re=710, 1.000461704532887
# Floquet unstable for a=0.38, Re=750, 1.0005028286053652
# Floquet unstable for a=0.38, Re=800, 1.0005410956036755
# Floquet unstable for a=0.38, Re=1000, 1.00058871226755

# play with time/space resolution:

# neumann bottom: Nt =1000, H=200, Nz=200
# Floquet stable for a=0.38, Re=500, 0.9988652350376159
# Floquet stable for a=0.38, Re=600, 0.9990750548925784


# neumann bottom: Nt =2000, H=200, Nz=200
# Floquet unstable for a=0.38, Re=600, 0.9990750548925768 
# Floquet unstable for a=0.38, Re=700, 0.9990750548925768 



# neumann bottom: Nt =1000, H=200, Nz=150
# Floquet stable for a=0.38, Re=1000, 0.9990898200962158

# neumann bottom: Nt =1000, H=300, Nz=200
# Floquet ?stable for a=0.38, Re=1000, 0.999091675520637


# dirchlet bottom: Nt =1000, H=100, Nz=100 
# Floquet stable for a=0.38, Re=400
# Floquet stable for a=0.38, Re=1000
# Floquet stable for a=0.38, Re=2000
# Floquet stable for a=0.38, Re=5000

# thom bottom: Nt =1000, H=100, Nz=100 
# Floquet ?stable for a=0.38, Re=400... numerical issues

# open bottom: most unstable at low Re (???) Nt =1000, H=100, Nz=100 
# Floquet unstable for a=0.38, Re=400 : mod 94
# Floquet unstable for a=0.38, Re=1000 : mod 6

     

print('Maximum modulus: ',maxmod)
if S == 1.:
  print('Floquet stable')
else:
  print('Floquet unstable')


