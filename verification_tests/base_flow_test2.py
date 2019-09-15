
import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy   
from scipy import signal
import sys
sys.path.insert(0, '/path/to/application/app/folder')
import functions as fn

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})

figure_path = "./figures/"
figure_path = './verification_tests/figures/base_flow_test/'

damper_scale = 100. # off if set to 0.
spur_damper = np.inf # off if np.inf
phi_path = '/home/bryan/git_repos/Floquet/figures/phi/'
psi_path = '/home/bryan/git_repos/Floquet/figures/psi/'


# =============================================================================

def stokes_solution( params, time, order ):
 # all dimensional: 
 zd = params['z']*params['dS'] # m, dimensional grid, zmax ~ Hd
 #z = params['z']
 #timed = time #*params['Td']/(2.*np.pi) # s, dimensional time
 # time goes [0,2pi]
 U = params['U']
 #time = params['
 omg = params['omg'] # omg is 1
 Nz = params['Nz']
 nu = params['nu']
 #L = params['L']
 Re = params['Re']
 dS = params['dS']
 #dS = np.sqrt(2.*nu/omg)
 #print(omg*timed)
 #print(omg*timed)
 u = U * np.exp( -zd/dS ) * np.cos( time - zd/dS )
 uz =  U/dS * np.exp( -zd/dS ) * ( np.sin( time - zd/dS ) - np.cos( time - zd/dS ) )
 uzz = - 2.*U/(dS**2.) * np.exp( -zd/dS ) * np.sin( time - zd/dS ) 
 """
 u = U * np.exp( -z ) * np.cos( omg*timed - z )
 uz =  U/dS * np.exp( -z ) * ( np.sin( omg*timed - z ) - np.cos( omg*timed - z ) )
 uzz = - 2.*U/(dS**2.) * np.exp( -z ) * np.sin( omg*timed - z ) 
 """
 if order < 1:
   return np.real(u)
 if order == 1:
   return np.real(u), np.real(uz)
 if order == 2:
   return np.real(u), np.real(uz), np.real(uzz)



def rotating_solution( params, time, order ):
 # phase: 
 # farfield b = sin(time)
 # farfield u = cos(time)
 
 # time = [0,2pi], non-dimensional (radians) time.
 # alternative: do time = dimensional time * omega

 # all dimensional: 
 z = params['z']*params['dS'] # m, dimensional grid 
 #timed = time*params['Td']/(2.*np.pi) # s, dimensional time
 U = params['U']
 N = params['N'] 
 omg = params['omg']
 Nz = params['Nz']
 Pr = params['Pr']
 nu = params['nu']
 kap = params['kap']
 L = params['L'] # U/omg
 f = params['f'] 
 tht = params['tht']
 Re = params['Re']
 #print(np.shape(tht),np.shape(Re))
 
 sqfcos = ( f * np.cos( tht ) )**2.
 sqNsin = ( N * np.sin( tht ) )**2.

 A = L*(omg**2. - sqfcos - sqNsin )
 
 # characteristic equation coefficients:
 a4 = - ( 2.*omg/nu + omg/kap ) 
 a2 = - (omg/nu)**2. - 2.*omg**2./(nu*kap) + (f*np.cos(tht)/nu)**2. + (N*np.sin(tht))**2./(nu*kap) 
 a0 =  omg / kap * ( (omg/nu)**2. - (f*np.cos(tht)/nu)**2. - (N*np.sin(tht)/nu)**2. ) 
 ap = ( A * N**2. * np.sin(tht) ) / ( omg**2. - sqfcos - sqNsin ) 

 # solution coefficients
 beta = ( -27.*1j*a0 + 9.*1j*a2*a4 + 2.*1j*(a4**3.) + \
           3.*np.sqrt(3.)*np.sqrt( -27.*a0**2. + 4.*a2**3. + \
           18.*a0*a2*a4 + a2**2.*a4**2. + 4.*a0*a4**3. ) )**(1./3.)
 #print(np.shape(beta),np.shape(a0))
 phi1 = beta/(3.*2.**(1./3.)) - (2.)**(1./3.) * (a4**2. + 3.*a2) / (3.*beta) - 1j*a4/3. 
 phi2 = - (1. - 1j*np.sqrt(3.)) * beta / (6.*2.**(1./3.)) +  \
          (1. + 1j*np.sqrt(3.)) * (a4**2. + 3.*a2) / (3.*2**(2./3.)*beta) - 1j*a4/3. 
 phi3 = - (1. + 1j*np.sqrt(3.)) * beta / (6.*2.**(1./3.)) +  \
         (1. - 1j*np.sqrt(3.)) * (a4**2. + 3.*a2) / (3.*2**(2./3.)*beta) - 1j*a4/3. 

 # ADD FUNCTION HERE THAT DETERMINES IF THE GRID IS APPROPRIATE
 """
 ups = kap**2.*nu*( np.sqrt(phi2*phi3)*phi1 + np.sqrt(phi1*phi3)*phi2 + \
                    np.sqrt(phi1*phi2)*phi3 ) + 1j*kap*nu*omg*( np.sqrt(phi1*phi2) + \
                    np.sqrt(phi1*phi3) + np.sqrt(phi2*phi3) + phi1 + phi2 + phi3 ) + \
                    nu*omg**2. + kap*(N*np.sin(tht))**2.
 
 c2 = - ( 1./( ups*(np.sqrt(phi1)-np.sqrt(phi2))*(np.sqrt(phi1)-np.sqrt(phi3)) ) ) * ( \
        A*kap*N**2.*np.sqrt(phi2*phi3)*np.sin(tht) + \
        1j*A*N**2.*omg*np.sin(tht) + \
        kap*N**2.*ap*np.sqrt(phi2*phi3)*np.sin(tht)**2. + \
        1j*kap*nu*ap*omg*( np.sqrt(phi3)*phi2**(3./2.) + phi3*phi2 + np.sqrt(phi2)*phi3**(3./2.) ) + \
        nu*ap*omg**2.*np.sqrt(phi2*phi3) )


 c4 =   ( 1./( ups*(np.sqrt(phi1)-np.sqrt(phi2))*(np.sqrt(phi2)-np.sqrt(phi3)) ) ) * ( \
        A*kap*N**2.*np.sqrt(phi1*phi3)*np.sin(tht) + \
        1j*A*N**2.*omg*np.sin(tht) + \
        kap*N**2.*ap*np.sqrt(phi1*phi3)*np.sin(tht)**2. + \
        1j*kap*nu*ap*omg*( np.sqrt(phi3)*phi1**(3./2.) + phi3*phi1 + np.sqrt(phi1)*phi3**(3./2.) ) + \
        nu*ap*omg**2.*np.sqrt(phi1*phi3) )


 c6 = - ( 1./( ups*(np.sqrt(phi1)-np.sqrt(phi3))*(np.sqrt(phi2)-np.sqrt(phi3)) ) ) * ( \
        A*kap*N**2.*np.sqrt(phi1*phi2)*np.sin(tht) + \
        1j*A*N**2.*omg*np.sin(tht) + \
        kap*N**2.*ap*np.sqrt(phi1*phi2)*np.sin(tht)**2. + \
        1j*kap*nu*ap*omg*( np.sqrt(phi2)*phi1**(3./2.) + phi2*phi1 + np.sqrt(phi1)*phi2**(3./2.) ) + \
        nu*ap*omg**2.*np.sqrt(phi1*phi2) )
 """

 E = np.zeros([3,3],dtype= complex)
 #print(np.shape(phi1))
 E[0,0] = omg + 1j*kap*phi1
 E[0,1] = omg + 1j*kap*phi2
 E[0,2] = omg + 1j*kap*phi3
 E[1,0] = 1j*omg*phi1*(kap+nu) - kap*nu*phi1**2. - (N*np.sin(tht))**2. + omg**2.
 E[1,1] = 1j*omg*phi2*(kap+nu) - kap*nu*phi2**2. - (N*np.sin(tht))**2. + omg**2.
 E[1,2] = 1j*omg*phi3*(kap+nu) - kap*nu*phi3**2. - (N*np.sin(tht))**2. + omg**2.
 E[2,0] = -np.sqrt(phi1)
 E[2,1] = -np.sqrt(phi2)
 E[2,2] = -np.sqrt(phi3)

 Y = np.zeros([3,1],dtype= complex)
 Y[0] = -omg*ap
 Y[1] = A*N**2.*np.sin(tht) - ap*( omg**2. - (N*np.sin(tht))**2. )
 Y[2] = 0.
 C = np.dot(np.linalg.inv(E),Y)
 c2 = C[0] # solution coefficient
 c4 = C[1] # solution coefficient
 c6 = C[2] # solution coefficient

 u1 = c2*(omg + 1j*kap*phi1)
 u2 = c4*(omg + 1j*kap*phi2)
 u3 = c6*(omg + 1j*kap*phi3)

 v1 = c2*(1j*omg*phi1*(kap+nu)-kap*nu*phi1**2.-(N*np.sin(tht))**2.+omg**2.)
 v2 = c4*(1j*omg*phi2*(kap+nu)-kap*nu*phi2**2.-(N*np.sin(tht))**2.+omg**2.)
 v3 = c6*(1j*omg*phi3*(kap+nu)-kap*nu*phi3**2.-(N*np.sin(tht))**2.+omg**2.) 

 # boundary condition checks:
 if abs(np.real(u1+u2+u3+ap*omg)) >= 1e-16:
  print('Wall boundary condition failure for u(z,t)')
  return
 if abs(np.real(v1+v2+v3-A*N**2.*np.sin(tht) + ap*(omg**2.-(N*np.sin(tht))**2) ) ) >= 1e-16:
  print('Wall boundary condition failure for v(z,t)')
  return 
 if abs(np.real(-c2*np.sqrt(phi1)-c4*np.sqrt(phi2)-c6*np.sqrt(phi3))) >= 1e-16:
  print('Wall boundary condition failure for b(z,t)')
  return 
 
 b = np.zeros([Nz]); u = np.zeros([Nz]); v = np.zeros([Nz]); 
 if order >= 1:
   bz = np.zeros([Nz]); uz = np.zeros([Nz]); vz = np.zeros([Nz]);
 if order == 2:
   bzz = np.zeros([Nz]); uzz = np.zeros([Nz]); vzz = np.zeros([Nz]);

 for i in range(0,Nz):
   u[i] = np.real( ( u1*np.exp(-np.sqrt(phi1)*z[i]) + u2*np.exp(-np.sqrt(phi2)*z[i]) + \
                     u3*np.exp(-np.sqrt(phi3)*z[i]) + ap*omg ) * np.exp(1j*time) ) / (N**2.*np.sin(tht))
   if order >= 1:
     uz[i] = np.real( ( - np.sqrt(phi1)*u1*np.exp(-np.sqrt(phi1)*z[i]) - np.sqrt(phi2)*u2*np.exp(-np.sqrt(phi2)*z[i]) - \
                        np.sqrt(phi3)*u3*np.exp(-np.sqrt(phi3)*z[i]) ) * np.exp(1j*time) ) / (N**2.*np.sin(tht))
   if order == 2:
     uzz[i] = np.real( ( np.sqrt(phi1)**2.*u1*np.exp(-np.sqrt(phi1)*z[i]) + np.sqrt(phi2)**2.*u2*np.exp(-np.sqrt(phi2)*z[i]) + \
                     + np.sqrt(phi3)**2.*u3*np.exp(-np.sqrt(phi3)*z[i]) ) * np.exp(1j*time) ) / (N**2.*np.sin(tht))
   if f > 0.:
     v[i] = np.real( ( v1*np.exp(-np.sqrt(phi1)*z[i]) + v2*np.exp(-np.sqrt(phi2)*z[i]) + \
                       v3*np.exp(-np.sqrt(phi3)*z[i]) + ap*(omg**2.-(N*np.sin(tht))**2 ) - \
                       A*N**2.*np.sin(tht) ) * 1j * np.exp(1j*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 
     if order >= 1:
       vz[i] = np.real( ( - np.sqrt(phi1)*v1*np.exp(-np.sqrt(phi1)*z[i]) - np.sqrt(phi2)*v2*np.exp(-np.sqrt(phi2)*z[i]) - \
                          np.sqrt(phi3)*v3*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 
     if order == 2:
       vzz[i] = np.real( ( np.sqrt(phi1)**2.*v1*np.exp(-np.sqrt(phi1)*z[i]) + np.sqrt(phi2)**2.*v2*np.exp(-np.sqrt(phi2)*z[i]) + \
                         np.sqrt(phi3)**2.*v3*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 
   if f <= 0.:
     v[i] = 0.
     if order >= 1:
       vz[i] = 0.
     if order == 2:
       vzz[i] = 0.
   b[i] = np.real( ( c2*np.exp(-np.sqrt(phi1)*z[i]) + c4*np.exp(-np.sqrt(phi2)*z[i]) + \
                     c6*np.exp(-np.sqrt(phi3)*z[i]) + ap ) * 1j * np.exp(1j*time) )  #omg*timed) ) 
   if order >= 1:
     bz[i] = np.real( ( - np.sqrt(phi1)*c2*np.exp(-np.sqrt(phi1)*z[i]) - np.sqrt(phi2)*c4*np.exp(-np.sqrt(phi2)*z[i]) - \
                        np.sqrt(phi3)*c6*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*time) )  #omg*timed) ) 
   if order == 2:
     bzz[i] = np.real( ( np.sqrt(phi1)**2.*c2*np.exp(-np.sqrt(phi1)*z[i]) + np.sqrt(phi2)**2.*c4*np.exp(-np.sqrt(phi2)*z[i]) + \
                      np.sqrt(phi3)**2.*c6*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*time) )  #omg*timed) )  
 
 if params['wall_flag'] == 'moving':
   u = u - np.real(U*np.exp(1j*time))  #omg*timed) ) 
   b = b - np.real(ap*1j*np.exp(1j*time))  #omg*timed) )  # why is this the opposite sign from the non-rotating case? Which one is wrong?
   if f > 0.:
     v = v - np.real( ( ap*(omg**2.-(N*np.sin(tht))**2 ) - A*N**2.*np.sin(tht) ) * 1j * np.exp(1j*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht))
     # subtract the geostropphic component of v; means the wall moves in the along-slope direction

 # dimensional output:
 if order < 1:
   return np.real(b), np.real(u), np.real(v)
 if order == 1:
   return np.real(b), np.real(u), np.real(v), np.real(bz), np.real(uz), np.real(vz)
 if order == 2:
   return np.real(b), np.real(u), np.real(v), np.real(bz), np.real(uz), np.real(vz), np.real(bzz), np.real(uzz), np.real(vzz)



def time_step( params , dt, stop_time, case_flag ):
  # uniform time step 4th-order Runge-Kutta time stepper

  time = 0. # non-dimensional time
  count = 0
  output_period = 10
  output_count = 0
  
  #start_time_0 = datetime.now()
  while time < stop_time - dt: 
    
    # plot base flow
    if case_flag == 'stokes':
        U,Uz,Uzz = stokes_solution( params, time, 2 ) # dimensional solutions (input time is [0,2pi])
        U = U / params['U'] # non-dimensional
        Uzz = Uzz / (params['U']) * params['dS']**2. # non-dimensional
        if params['plot_freq'] != 0:
            pfreq = params['plot_freq']
            if np.floor(count/pfreq) == count/pfreq:
                stokes_plot( U, Uzz, time, params['z']/params['H'], count )

    if case_flag == 'abyss':
        B, U, V, Bz, Uz, Vz, Bzz, Uzz, Vzz = rotating_solution( params, time, 2 )
        U = U / params['U'] # non-dimensional
        Uzz = Uzz / (params['U']) * params['dS']**2. # non-dimensional
        Bz = Bz * (params['dS']*params['omg']) / ( (params['N'])**2 * params['U'] )  # non-dimensional
        if params['plot_freq'] != 0:
            pfreq = params['plot_freq']
            if np.floor(count/pfreq) == count/pfreq:
                abyss_plot( U, Uzz, Bz, time, params['z']/params['H'], count )

    time = time + dt # non-dimensional time
    count = count + 1

  dtf = stop_time - time
  final_time = time + dtf # non-dimensional final time

  # plot base flow at final time 
  if case_flag == 'stokes':
      U,Uz,Uzz = stokes_solution( params, time, 2 ) # dimensional solutions (input time is [0,2pi])
      U = U / params['U'] # non-dimensional
      Uzz = Uzz / (params['U']) * params['dS']**2. # non-dimensional
      if params['plot_freq'] != 0:
          pfreq = params['plot_freq']
          if np.floor(count/pfreq) == count/pfreq:
              stokes_plot( U, Uzz, time, params['z']/params['H'], count )

  if case_flag == 'abyss':
      B, U, V, Bz, Uz, Vz, Bzz, Uzz, Vzz = rotating_solution( params, time, 2 )
      U = U / params['U'] # non-dimensional
      Uzz = Uzz / (params['U']) * params['dS']**2. # non-dimensional
      Bz = Bz * (params['dS']*params['omg']) / ( (params['N'])**2 * params['U'] )  # non-dimensional
      if params['plot_freq'] != 0:
          pfreq = params['plot_freq']
          if np.floor(count/pfreq) == count/pfreq:
              abyss_plot( U, Uzz, Bz, time, params['z']/params['H'], count )


  count = count + 1

  return final_time

def stokes_plot( U, Uzz, time, z, count ):

    plotname = figure_path + '%i.png' %(count)
    fig = plt.figure(figsize=(10,5)); 
    plt.subplot(1,2,1)
    plt.plot(U,z,color='royalblue',linewidth=2.) 
    plt.xlabel(r"$u/U_\infty$",fontsize=16)
    plottitle = r"$t/T=%.3f$" %(time/(2.*np.pi))
    plt.title(plottitle,fontsize=16)
    plt.ylabel(r"$z/H$",fontsize=16)
    plt.xlim([-1.1,1.1])
    plt.grid();
    plt.subplot(1,2,2)
    plt.plot(Uzz,z,color='royalblue',linewidth=2.) 
    plt.xlabel(r"$u_{zz}\delta^2/U_\infty$",fontsize=16)
    plt.ylabel(r"$z/H$",fontsize=16); plt.grid();
    plt.xlim([-2.,2.])
    plottitle = r"$t/T=%.3f$" %(time/(2.*np.pi))
    plt.title(plottitle,fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.125, left=0.075, right=0.98, hspace=0.08, wspace=0.2)
    plt.savefig(plotname,format="png")
    plt.close(fig);

    return

def abyss_plot( U, Uzz, Bz, time, z, count ):

    plotname = figure_path + '%i.png' %(count)
    fig = plt.figure(figsize=(15,5)); 
    plt.subplot(1,3,1)
    plt.plot(U,z,color='royalblue',linewidth=2.) 
    plt.xlabel(r"$u/U_\infty$",fontsize=16)
    plottitle = r"$t/T=%.3f$" %(time/(2.*np.pi))
    plt.title(plottitle,fontsize=16)
    plt.ylabel(r"$z/H$",fontsize=16)
    plt.grid(); plt.xlim([-1.1,1.1])
    plt.subplot(1,3,2)
    plt.plot(Uzz,z,color='royalblue',linewidth=2.) 
    plt.xlabel(r"$u_{zz}\delta^2/U_\infty$",fontsize=16)
    plt.ylabel(r"$z/H$",fontsize=16); plt.grid();
    plottitle = r"$t/T=%.3f$" %(time/(2.*np.pi))
    plt.title(plottitle,fontsize=16)
    plt.xlim([-2.,2.])
    plt.subplot(1,3,3)
    plt.plot(Bz,z,color='crimson',linewidth=2.) 
    plt.xlabel(r"$b_{z}\omega\delta/(N^2U_\infty)$",fontsize=16)
    plt.ylabel(r"$z/H$",fontsize=16); plt.grid();
    plottitle = r"$t/T=%.3f$" %(time/(2.*np.pi))
    plt.title(plottitle,fontsize=16); plt.xlim([-0.02,0.02])
    plt.subplots_adjust(top=0.925, bottom=0.125, left=0.075, right=0.98, hspace=0.08, wspace=0.2)
    plt.savefig(plotname,format="png")
    plt.close(fig);

    return

# =============================================================================


T = 2.*np.pi # s, period
omg = 2.*np.pi/44700. # rads/s
nu = 1e-6
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness

Rej = np.array([781.4*2.])
ai = np.array([0.3]) 
#Rej = np.linspace(1300.,1400.,num=4,endpoint=True)
#ai = np.linspace(0.025,0.5,num=20,endpoint=True)

# grid
grid_flag = 'uniform' #'hybrid cosine' #'  'cosine' # # 
wall_BC_flag = 'BC'
plot_freq = 1000
Nz = 200 # 
H = 32. # = Hd/dS, non-dimensional grid height
CFL = 2. # 
Hd = H*dS # m, dimensional domain height (arbitrary choice)
z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid

# abyss:
N = 1e-3 # 1/s, buoyancy frequency
f = 0. #1e-4 # 1/s, inertial frequency
C = 0.25
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
wall_flag = 'moving'

# pre-constructed matrices:
grid_params_dzz = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
grid_params_inv = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex ) # identity matrix
dzz_zeta,lBC = fn.diff_matrix( grid_params_dzz , 'dirchlet 2' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
dzz_psi,lBC2 = fn.diff_matrix( grid_params_inv , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
dzz_zeta = np.multiply(dzz_zeta,np.ones(np.shape(dzz_zeta)),dtype=complex) 
dzz_psi = np.multiply(dzz_psi,np.ones(np.shape(dzz_psi)),dtype=complex)
A0 = np.zeros( [Nz,Nz] , dtype=complex ) # initial propogator matrix 

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
        Nt = 50000 #int(2.*np.pi/dt)
        freq = int(Nt/100)
        print('number of time steps, Nt = ',Nt)

        inv_psi = np.linalg.inv( dzz_psi - (a**2.*eye_matrix) ) 
        inv_psi = np.multiply(inv_psi,np.ones(np.shape(inv_psi)),dtype=complex)

        # parameters for monodromy matrix computation:
        params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U, 'inv_psi':inv_psi, 'plot_freq':plot_freq, 'grid_flag':grid_flag,
          'Nz':Nz, 'Nt':Nt, 'Re':Re,'a':a, 'H':H, 'Hd':Hd, 'dzz_zeta':dzz_zeta, 'CFL':CFL, 'A0':A0, 'damper_scale':damper_scale, 
          'spur_damper':spur_damper, 'Pr':Pr, 'tht':tht, 'N':N, 'f':f, 'kap':kap, 'L':U/omg, 'wall_flag':wall_flag, 'C2':C**2.,
          'dS':dS, 'z':z, 'dz':dz, 'eye_matrix':eye_matrix,'freq':freq, 'lBC':lBC, 'lBC2':lBC2, 'phi_path':phi_path, 'psi_path':psi_path} 
        Nc = fn.count_points( params )
        print('number of points within delta = %i' %(Nc))

        final_time = time_step( params, T/Nt, T , 'abyss' )

















"""

T = 2.*np.pi # radians, non-dimensional period
Td = 44700. # s, M2 tide period

Nz = 200 # number of grid points
grid_flag = 'cosine' # 'uniform'
wall_flag = 'moving' 
#wall_flag = 'farfield' 

nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
omg = 2.0*np.pi/Td # rads/s
f = 0. #1e-4 # 1/s, inertial frequency
N = 1e-3 # 1/s, buoyancy frequency
C = 1./4. # N^2*sin(tht)/omg, slope ``criticality''
U = 0.01 # m/s, oscillation velocity amplitude
L = U/omg # m, excursion length (here we've assumed L>>H)
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians
Re = omg*L**2./nu # Reynolds number
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
ReS = np.sqrt(2.*Re) # Stokes' 2nd problem Reynolds number
H = 1.
Hd = 100.*dS # m, dimensional domain height 
z,dz = fn.grid_choice( grid_flag , Nz , H )

Nt = 100 # number of time steps
dt = T/(Nt-1) # non-dimensional dt

params = {'nu': nu, 'kap': kap, 'omg': omg, 'L':L, 'T': T, 'Td': Td, 'U': U, 'H': H, 'Hd': Hd,
          'N':N, 'tht':tht, 'Re':Re, 'C':C, 'H':H, 'Nz':Nz, 'wall':wall_flag,
          'dS':dS, 'ReS':ReS, 'thtc':thtc, 'grid':grid_flag, 'f': f, 'Pr':Pr,
          'dz_min':(np.amin(dz)),'Nt':Nt, 'CFL':(dt/np.amin(dz)), 'z':z, 'dz':dz}

"""

# non-dimensionalization consistent with Blennerhasset:




# =============================================================================






"""
# non-rotating solutions
u = np.zeros([Nz,Nt]); b = np.zeros([Nz,Nt]); 
uz = np.zeros([Nz,Nt]); bz = np.zeros([Nz,Nt]); 

# rotating solutions
ur = np.zeros([Nz,Nt]); vr = np.zeros([Nz,Nt]); br = np.zeros([Nz,Nt]); 
uzr = np.zeros([Nz,Nt]); vzr = np.zeros([Nz,Nt]); bzr = np.zeros([Nz,Nt]); 
uzzr = np.zeros([Nz,Nt]); vzzr = np.zeros([Nz,Nt]); bzzr = np.zeros([Nz,Nt]);

# verification of the finite differencing
uz_check = np.zeros([Nz,Nt]); bz_check = np.zeros([Nz,Nt]);
uzz_check = np.zeros([Nz,Nt]); bzz_check = np.zeros([Nz,Nt]);

uzr_check = np.zeros([Nz,Nt]); bzr_check = np.zeros([Nz,Nt]);
uzzr_check = np.zeros([Nz,Nt]); bzzr_check = np.zeros([Nz,Nt]);
"""
