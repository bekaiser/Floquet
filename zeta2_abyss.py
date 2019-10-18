
# make sure the A dot phi operation is correct

# fix wavenumber indexing

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
stat_path = "./output/"
email_flag = 0
ic_plot_flag = 0
damper_scale = 100. # off if set to 0.
spur_damper = np.inf # off if np.inf
zeta_path = '/home/bryan/git_repos/Floquet/figures/zeta/'
psi_path = '/home/bryan/git_repos/Floquet/figures/psi/'
b_path = '/home/bryan/git_repos/Floquet/figures/b/'


if email_flag == 1:
    import smtplib, ssl
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "my.sent.data.kaiser@gmail.com"  # Enter your address
    receiver_email = "my.sent.data.kaiser@gmail.com"  # Enter receiver address
    password = 'wh$0i1BTu' #input("Type your password and press enter: ")


# =============================================================================
# need a resolution requirement. From the analytical solution?

T = 2.*np.pi # s, period
omg = 2.*np.pi/44700. # rads/s
nu = 1e-6
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness

Rej = np.array([1400.])
ai = np.array([0.35]) 
#Rej = np.linspace(1300.,1400.,num=4,endpoint=True)
#ai = np.linspace(0.025,0.5,num=20,endpoint=True)

# abyss:
N = 1e-3 # 1/s, buoyancy frequency
C = 0.01 # try for C = 0.01
f = 0.
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
wall_flag = 'moving'

# grid
grid_flag = 'uniform' #'hybrid cosine' #'  'cosine' # # 
wall_BC_flag = 'BC'
plot_freq = 1000
Nz = 200 # 
H = 32. # = Hd/dS, non-dimensional grid height
CFL = 2. # 
Hd = H*dS # m, dimensional domain height (arbitrary choice)
z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid

# pre-constructed matrices:
grid_params_dzz = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
grid_params_inv = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
grid_params_b = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':' '} 
eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex ) # identity matrix
dzz_zeta,lBC = fn.diff_matrix( grid_params_dzz , 'dirchlet 2' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
dzz_psi,lBC2 = fn.diff_matrix( grid_params_inv , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
dzz_b = fn.diff_matrix( grid_params_b , 'neumann' , 'neumann' , diff_order=2 , stencil_size=3 ) 
dz_b = fn.diff_matrix( grid_params_b , 'neumann' , 'neumann' , diff_order=1 , stencil_size=3 ) 

dzz_zeta = np.multiply(dzz_zeta,np.ones(np.shape(dzz_zeta)),dtype=complex) 
dzz_psi = np.multiply(dzz_psi,np.ones(np.shape(dzz_psi)),dtype=complex)
dzz_b = np.multiply(dzz_zeta,np.ones(np.shape(dzz_b)),dtype=complex) 
dz_b = np.multiply(dzz_zeta,np.ones(np.shape(dz_b)),dtype=complex)

A0 = np.zeros([int(2*Nz),int(2*Nz)],dtype=complex) # initial propogator matrix
#A0 = np.zeros( [Nz,Nz] , dtype=complex )  

Nj = np.shape(Rej)[0]; Ni = np.shape(ai)[0]
M = np.zeros([Nj,Ni]); Mr = np.zeros([Nj,Ni]); Mi = np.zeros([Nj,Ni]);

print('\nGrid:',grid_flag)
print('Nz/H:',Nz/H)

for i in range(0,Ni):
    for j in range(0,Nj):

        print('\nReynolds number: %.1f' %(Rej[j]) )
        print('disturbance wavenumber: %.2f' %(ai[i]) )
        print('H: %.1f' %(H), 'Nz: %i' %(Nz), 'CFL: %.2f' %(CFL) )

        Re = Rej[j]
        a = ai[i]
        U = Re * (nu/dS) # Re = U*dS/nu, so ReB=Re/2
        dt = CFL*(z[0]/Re)  # = CFL*(np.amin(dz)/Re) 
        Nt = int(2.*np.pi/dt)
        if Nt < 10000:
            Nt = 10000
        freq = int(Nt/10)
        print('number of time steps, Nt = ',Nt)

        inv_psi = np.linalg.inv( dzz_psi - (a**2.*eye_matrix) ) 
        inv_psi = np.multiply(inv_psi,np.ones(np.shape(inv_psi)),dtype=complex)

        # parameters for monodromy matrix computation:
        params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U, 'inv_psi':inv_psi, 'plot_freq':plot_freq, 'grid_flag':grid_flag,
          'Nz':Nz, 'Nt':Nt, 'Re':Re,'a':a, 'H':H, 'Hd':Hd, 'dzz_zeta':dzz_zeta, 'CFL':CFL, 'A0':A0, 'damper_scale':damper_scale, 
          'spur_damper':spur_damper, 'Pr':Pr, 'tht':tht, 'N':N, 'f':f, 'kap':kap, 'L':U/omg, 'wall_flag':wall_flag, 
          'dzz_b':dzz_b, 'dz_b':dz_b, 'C2':((N/omg)**2.), 'b_path':b_path, 'zeta_path':zeta_path, 
          'psi_path':psi_path, 'stat_path':stat_path, 
          'dS':dS, 'z':z, 'dz':dz, 'eye_matrix':eye_matrix,'freq':freq, 'lBC':lBC, 'lBC2':lBC2} 
        Nc = fn.count_points( params )
        print('number of points within delta = %i' %(Nc))

        # initial conditions (prinicipal fundamental solution matrix):
        Phi0 = np.eye(int(2*Nz),int(2*Nz),0,dtype=complex)

        # correct:
        #Bn = Phi0[Nz:int(2*Nz),:] # zeta
        #Zn = Phi0[0:Nz,:] # buoyancy
        #Pn = np.real(np.dot(params['inv_psi'],Zn)) # psi

        # compute monodromy matrix:
        Phin,final_time = fn.rk4_time_step( params, Phi0 , T/Nt, T , 'zeta2' ) # spanwise vorticity

        # store maxima:
        Fmult = np.linalg.eigvals(Phin) 
        # find locations where imag >= 0.5, then zero out those locations:
        # Fmult2 = fn.remove_high_frequency( Fmult )
      
        print(Fmult)      
 
        M[j,i] = np.amax(np.abs(Fmult)) # maximum modulus, eigenvals = floquet multipliers
        Mr[j,i] = np.amax(np.real(Fmult))
        Mi[j,i] = np.amax(np.imag(Fmult))
        print('\nmax. modulus mu = ',M[j,i])
        print('\nmax. real mu = ',Mr[j,i])      
        print('\nmax. imag mu = ',Mi[j,i]) 
        print('\n')

        # output file with all multipliers, not just maxima:
        h5_filename = stat_path + "multiplier_Re%i_a%i.h5" %(Rej[j],int(ai[i]*1000))
        f2 = h5py.File(h5_filename, "w")
        dset = f2.create_dataset('CFL', data=CFL, dtype='f8')
        dset = f2.create_dataset('Nz', data=Nz, dtype='f8')
        dset = f2.create_dataset('H', data=H, dtype='f8')
        dset = f2.create_dataset('multR', data=np.real(Fmult), dtype='f8')
        dset = f2.create_dataset('multI', data=np.imag(Fmult), dtype='f8')

        # email with all multipliers, not just maxima:
        if email_flag == 1:
            Fmult = np.array2string(Fmult, precision=6, separator=',',suppress_small=True)
            Fmult_Psi = np.array2string(Fmult_Psi, precision=6, separator=',',suppress_small=True)
            message = """\
            Subject: data k = %.3f Re = %.1f

            Nz = %i\n
            CFL = %.3f\n
            maximum modulus zeta = %.3f\n 
            maximum modulus psi = %.3f\n zeta multiplier:\n""" %(ai[i],Rej[j],Nz,CFL,M[j,i],MP[j,i])
            message = message + Fmult + """\n psi multiplier:\n""" + Fmult_Psi
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, message)




