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

def count_points( params ):
    #dS = params['dS']
    z = params['z']
    Nz = params['Nz']
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

Ngrid = 2 
#Rej = np.array([3000])
#ai = np.array([0.36666666666666666])
Rej = np.linspace(500,3000,num=Ngrid,endpoint=True)
ai = np.linspace(0.05,1.,num=Ngrid,endpoint=True)

# grid
grid_flag = 'tanh' # 'uniform' #'  'cosine' # # 
wall_flag = 'moving'
H = 500. # = Hd/dS, non-dimensional grid height
Hd = H*dS # m, dimensional domain height (arbitrary choice)
CFL = 1.

M = np.zeros([Ngrid,Ngrid,3]);
Mr = np.zeros([Ngrid,Ngrid,3]);
Mi = np.zeros([Ngrid,Ngrid,3]);

for i in range(0,Ngrid):
    for j in range(0,Ngrid):

        print('\nReynolds number: %.1f' %(Rej[j]) )
        print('disturbance wavenumber: %.2f' %(ai[i]) )

        Re = Rej[j]
        a = ai[i]
        U = Re * (nu/dS) # Re = U*dS/nu, so ReB=Re/2

        Nzn = np.array([33,45])

        start_time_ij = datetime.now()

        for n in range(0,2):

            Nz = Nzn[n]
            z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid
            grid_params = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz} 
            # dzz_zeta: could try neumann LBC. Upper BC irrotational (no-stress).
            dzz_zeta = fn.diff_matrix( grid_params , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) # non-dimensional
            # inv_psi: lower BCs are no-slip, impermiable, upper BC is impermiable, free-slip
            inv_psi = np.linalg.inv( fn.diff_matrix( grid_params , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 ) ) # non-dimensional
            eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex )

            dt = CFL*(np.amin(dz)/Re) 
            Nt = int(2.*np.pi/dt)
            print('Nz: %i' %(Nz), 'Nt: %i' %(Nt) )     
            freq = int(Nt/5)

            params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U, 'inv_psi':inv_psi,  
            'Nz':Nz, 'Nt':Nt, 'Re':Re,'a':a, 'H':H, 'Hd':Hd, 'dzz_zeta':dzz_zeta,
            'dS':dS, 'z':z, 'dz':dz, 'eye_matrix':eye_matrix,'freq':freq} 

            Phi0 = np.eye(int(Nz),int(Nz),0,dtype=complex) # initial condition (prinicipal fundamental solution matrix)
            Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'blennerhassett' )
            Fmult = np.linalg.eigvals(Phin)
            M[j,i,n] = np.amax(np.abs(Fmult)) # maximum modulus, eigenvals = floquet multipliers
            Mr[j,i,n] = np.amax(np.real(Fmult))
            Mi[j,i,n] = np.amax(np.imag(Fmult))
            print('\nmaximum modulus = ',M[j,i,n])
            print('\nmaximum real mu = ',Mr[j,i,n])      
            print('\nmaximum imag mu = ',Mi[j,i,n]) 
            
        time_elapsed = datetime.now() - start_time_ij
        print('Wall time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

        R = 19./8. # refinement ratio
        P = 2. # order of convergence
        M[j,i,2] = M[j,i,1] + (M[j,i,1] - M[j,i,0]) / ( R**P - 1. ) 
        Mr[j,i,2] = Mr[j,i,1] + (Mr[j,i,1] - Mr[j,i,0]) / ( R**P - 1. ) 
        Mi[j,i,2] = Mi[j,i,1] + (Mi[j,i,1] - Mi[j,i,0]) / ( R**P - 1. ) 
        print('\nExtrapolated maximum modulus = ',M[j,i,2])
        print('\nExtrapolated maximum real mu = ',Mr[j,i,2])      
        print('\nExtrapolated maximum imag mu = ',Mi[j,i,2]) 


aI,ReJ = np.meshgrid(ai,Rej)

    
plotname = figure_path +'strutt.png' 
plottitle = r"maximum modulus" #, $N_z$ = %i, $H/\delta_S$ = %.1f" %(int(Nz),H) #r"Re = %.1f a = %.2f Nt = %i" %(Re,a,Nt)
fig = plt.figure(figsize=(8, 8))
CS = plt.contourf(aI,ReJ,M[:,:,2],cmap='gist_gray')
plt.xlabel(r"k",fontsize=16);
plt.ylabel(r"Re",fontsize=16); 
plt.colorbar(CS)
plt.title(plottitle,fontsize=16);
plt.savefig(plotname,format="png"); plt.close(fig);


