#
# Bryan Kaiser

# find a convergence for high resolution of BL: keep domain height constant (minimum points in BL needed).
# then find a convergence for domain size: keep the BL res constant but vary the height.


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

def count_points( params ):
    dS = params['dS']
    z = params['z']
    Hd = params['Hd']
    Nz = params['Nz']
    count = 0
    for j in range(0,Nz):
         if (z[j]*Hd) <= dS:
             count = count + 1
    return count


 



# =============================================================================
# need a resolution requirement. From the analytical solution?

T = 2.*np.pi # s, period
omg = 2.*np.pi/T # rads/s
nu = 1e-6
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
#
#Nz = 400 # number of grid points

a = 0.38
Re = 800.
U = Re * (nu/dS) # Re = U*dS/nu, so ReB=Re/2
#print(U)
grid_flag = 'cosine' # 'uniform' 
wall_flag = 'moving'

#Nt = 5000


#Ngrid = 7
#Nzi = np.linspace(100,400,num=Ngrid,endpoint=True)
#Hj = np.linspace(10,50,num=Ngrid,endpoint=True)

Ngrid = 1
Nzi = [150]
Hj = [6.]

#S = np.zeros([Ngrid,Ngrid]);
M = np.zeros([Ngrid,Ngrid]);

for i in range(0,Ngrid):
    for j in range(0,Ngrid):

        print('\nDomain height (non-dimensional): %.1f' %(Hj[j]) )
        print('Resolution: %i' %(int(Nzi[i])) )
 
        Nz = int(Nzi[i])
        H = Hj[j] # = Hd/dS, non-dimensional grid height
        Hd = H*dS # m, dimensional domain height (arbitrary choice)
      
        # get relevant parameters / construct differntiation matrices
        z,dz = fn.grid_choice( 'cosine' , Nz , H ) # non-dimensional grid
        #print(dz[0]*Hd)
        grid_params = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz} 
        dzz_zeta = fn.diff_matrix( grid_params , 'neumann' , 'dirchlet' , diff_order=2 , stencil_size=3 ) # non-dimensional
        # dzz_zeta: could try neumann LBC. Upper BC irrotational (no-stress).
        inv_psi = np.linalg.inv( fn.diff_matrix( grid_params , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 ) ) # non-dimensional
        # inv_psi: lower BCs are no-slip, impermiable, upper BC is impermiable, free-slip
        eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex )

        CFL = 16.
        #dt1 = CFL*(np.amin(dz*Hd))*omg # s*rads/s = rads, non-dimensional dt
        #dt2 = CFL*((np.amin(dz*Hd))**2./nu)*omg # s*rads/s = rads, non-dimensional dt
        #dt1 = CFL*((dz[2]*Hd))*omg # s*rads/s = rads, non-dimensional dt
        #dt2 = CFL*(((dz[2]*Hd))**2./nu)*omg # s*rads/s = rads, non-dimensional dt
        #dt = np.amin([dt1,dt2])
        dt = CFL*(np.amin(dz*Hd)/U) # 
        """
        if dt == dt1:
            print('Advective CFL')
        if dt == dt2:
            print('Diffusive CFL')
        """
        Nt = int(T/dt)
        #print(Nt)
        #Nt = 2000 
        print('Number of time steps, Nt = ',Nt)
     
        freq = int(Nt/10)

        params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U, 'inv_psi':inv_psi,  
          'Nz':Nz, 'Nt':Nt, 'Re':Re,'a':a, 'H':H, 'Hd':Hd, 'dzz_zeta':dzz_zeta,
          'dS':dS, 'z':z, 'dz':dz, 'eye_matrix':eye_matrix,'freq':freq} 

        Nc = count_points( params )
        print('number of points within delta = %i' %(Nc))

        Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
        Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'blennerhassett' )
        mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
        M[j,i] = np.amax(mod)
        print('maximum modulus = ',M[j,i])
       
        """
        if M[j,i] <= 1.:
            S[j,i] = 1. # 1 is for stability
        if M[j,i] >= 1e10:
            S[j,i] = 10. # numerical problems
        """


"""
#print('Maximum modulus: ',maxmod)
if S == 1.:
  print('Floquet stable')
else:
  print('Floquet unstable')
"""
#print('Number of time steps, Nt = ',Nt)

#print(S)
print(M)
#print(a)
#print(Re)

NzI,HJ = np.meshgrid(Nzi,Hj)

"""     
plotname = figure_path +'convergence.png' 
plottitle = r"Re = %.1f a = %.2f Nt = %i" %(Re,a,Nt)
fig = plt.figure()
CS = plt.contourf(NzI,HJ,M,cmap='gist_gray')
plt.xlabel(r"$N_z$",fontsize=13);
plt.ylabel(r"$H/\delta$",fontsize=13); 
plt.colorbar(CS)
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);
"""

