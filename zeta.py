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

# thom bot bc on psi gives the same as dirchlet and neumann (for no base flow case)! Something is wrong! 
# eye matrix or ones somewhere?

# =============================================================================

def count_points( params ):
    #dS = params['dS']
    z = params['z']
    #Hd = params['Hd']
    #Nz = params['Nz']
    count = 0
    for j in range(0,Nz):
         #if (z[j]*dS) <= dS:
         if (z[j]) <= 1.:
             count = count + 1
    return count


# =============================================================================
# need a resolution requirement. From the analytical solution?

T = 2.*np.pi # s, period
omg = 2.*np.pi/44700. # rads/s
nu = 1e-6
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness

Ngrid = 1 #46
Rej = np.array([1000])
ai = np.array([0.3]) #36666666666666666])
#Rej = np.linspace(200,300,num=Ngrid,endpoint=True)
#ai = np.linspace(0.05,0.6,num=Ngrid,endpoint=True)

# grid
grid_flag = 'tanh' #'uniform' #'  'cosine' # # 
wall_BC_flag = 'Thom'
off_flag = ' '
Nz = 25
H = 500. # = Hd/dS, non-dimensional grid height
CFL = 0.5
#Nz = np.array([50,75,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650])
#H = np.array([2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.,22.,24.,26.])
Hd = H*dS # m, dimensional domain height (arbitrary choice)
z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid
grid_params_dzz = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
grid_params_inv = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':off_flag} 
# dzz_zeta: could try neumann LBC. Upper BC irrotational (no-stress).
dzz_zeta,lBC = fn.diff_matrix( grid_params_dzz , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) # non-dimensional
# inv_psi: lower BCs are no-slip, impermiable, upper BC is impermiable, free-slip
eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex ) #np.ones([Nz,Nz],dtype=complex) A BIG DIFFERENCE!

# non-dimensional
#eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex )

# all parts of the forcing need to be complex arrays:
dzz_zeta = np.multiply(dzz_zeta,np.ones(np.shape(dzz_zeta)),dtype=complex)
#inv_psi = np.multiply(inv_psi,np.ones(np.shape(inv_psi)),dtype=complex)
lBC = lBC + 0.j
#print(lBC)

M = np.zeros([Ngrid,Ngrid]);
Mr = np.zeros([Ngrid,Ngrid]);
Mi = np.zeros([Ngrid,Ngrid]);

#sf = 1000.

print('\nGrid:',grid_flag)
print('Nz/H:',Nz/H)
for i in range(0,Ngrid):
    for j in range(0,Ngrid):

        print('\nReynolds number: %.1f' %(Rej[j]) )
        print('disturbance wavenumber: %.2f' %(ai[i]) )
        print('H: %.1f' %(H), 'Nz: %i' %(Nz), 'CFL: %.2f' %(CFL) )

        Re = Rej[j]
        a = ai[i]
        U = Re * (nu/dS) # Re = U*dS/nu, so ReB=Re/2
    
        #dt = CFL*(np.amin(dz)/Re) 
        dt = CFL*(z[0]/Re) 
        Nt = int(2.*np.pi/dt)
      
        freq = int(Nt/100)
        print('number of time steps, Nt = ',Nt)

        dzz_psi = fn.diff_matrix( grid_params_inv , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 )
        dzz_psi = np.multiply(dzz_psi,np.ones(np.shape(dzz_psi)),dtype=complex)
        inv_psi = np.linalg.inv( dzz_psi - (a**2.*eye_matrix) ) 

        phi_path = '/home/bryan/Desktop/Floquet/figures/phi/'
        psi_path = '/home/bryan/Desktop/Floquet/figures/psi/'
        params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U, 'inv_psi':inv_psi,  
          'Nz':Nz, 'Nt':Nt, 'Re':Re,'a':a, 'H':H, 'Hd':Hd, 'dzz_zeta':dzz_zeta,
          'dS':dS, 'z':z, 'dz':dz, 'eye_matrix':eye_matrix,'freq':freq, 'lBC':lBC, 'phi_path':phi_path, 'psi_path':psi_path} 

        Nc = count_points( params )
        print('number of points within delta = %i' %(Nc))

        Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
        Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'blennerhassett' )
        Fmult = np.linalg.eigvals(Phin)
        M[j,i] = np.amax(np.abs(Fmult)) # maximum modulus, eigenvals = floquet multipliers
        Mr[j,i] = np.amax(np.real(Fmult))
        Mi[j,i] = np.amax(np.imag(Fmult))
        print('\nmaximum modulus = ',M[j,i])
        print('\nmaximum real mu = ',Mr[j,i])      
        print('\nmaximum imag mu = ',Mi[j,i]) 

        # add plots of psi final solutions

print('Reynolds number = ',Rej)
print('wavenumber = ', ai)
print('Nz = ',Nz)
print('CFL = ',CFL)
print('Nt = ',Nt)
print('Grid = ',grid_flag)
print('number of points within delta = %i' %(Nc))



