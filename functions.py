# functions for Floquet analysis

#import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
#from scipy.stats import chi2
#from scipy import signal
#from scipy.fftpack import fft, fftshift
#import matplotlib.patches as mpatches
#from matplotlib.colors import colorConverter as cc
from datetime import datetime
import numpy.distutils.system_info as sysinfo
sysinfo.get_info('atlas')


# =============================================================================    

def count_points( params ):
    #dS = params['dS']
    z = params['z']
    #Hd = params['Hd']
    #Nz = params['Nz']
    count = 0
    for j in range(0,params['Nz']):
         #if (z[j]*dS) <= dS:
         if (z[j]) <= 1.:
             count = count + 1
    return count

def remove_high_frequency( E ):
    G = np.copy(E)
    locs=np.argwhere(np.imag(G)>0.5)
    Nl = np.shape(locs)[0]
    for m in range(0,Nl):
        G[locs[m,:][0],locs[m,:][1]]=0.+0.j
    return G


# =============================================================================    
# time-step functions functions


def rk4_time_step( params , Phin , dt, stop_time, case_flag ):
  # uniform time step 4th-order Runge-Kutta time stepper

  time = 0. # non-dimensional time
  count = 0
  output_period = 10
  output_count = 0
  
  start_time_0 = datetime.now()
  while time < stop_time - dt: 

    #start_time_kcoeffs = datetime.now()
    k1 = rk4( params , time , Phin , count , 0 , case_flag )
    k2 = rk4( params , time + dt/2. , Phin + k1*dt/2. , count , 0 , case_flag )
    k3 = rk4( params , time + dt/2. , Phin + k2*dt/2. , count , 0 , case_flag )
    k4 = rk4( params , time + dt , Phin + k3*dt , count , 0 , case_flag )
    #time_elapsed = datetime.now() - start_time_kcoeffs
    #print('k coeff time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    #start_time_Phi_update = datetime.now()
    Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dt/6.; 
    #time_elapsed = datetime.now() - start_time_Phi_update
    #print('Phi update time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    #output_count = perturb_monitor( time , count , output_count , output_period , Phin , z , params , 'plot' )
    time = time + dt # non-dimensional time
    count = count + 1
    #freq = 100
    if params['freq'] != 0:
        if np.floor(count/params['freq']) == count/params['freq']:
            print( '%.2f complete' %(count/params['Nt']) )
            time_elapsed = datetime.now() - start_time_0
            print('Wall time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    check_matrix(Phin,'Phin')

  dtf = stop_time - time
  k1 = rk4( params , time , Phin , count , 0 , case_flag )
  k2 = rk4( params , time + dtf/2. , Phin + k1*dtf/2. , count , 0 , case_flag )
  k3 = rk4( params , time + dtf/2. , Phin + k2*dtf/2. , count , 0 , case_flag )
  k4 = rk4( params , time + dtf , Phin + k3*dtf , count , 0 , case_flag )
  Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dtf/6.; 

  # this is where conservation of mass needs to be checked

  final_time = time + dtf # non-dimensional final time
  count = count + 1

  if params['freq'] != 0:
      print( '%.2f complete' %(count/params['Nt']) )
      print('Wall time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

  #print('RK4 method, final time = ', final_time)
  return Phin, final_time


def rk4( params , time , Phin , count , plot_flag , case_flag ):
  # 4th-order Runge-Kutta functions 
  
  if case_flag == 'boundary_layer':
    # dimensional the base periodic flow:
    ( b, u, bz, uz ) = xforcing_nonrotating_solution( params['U'], params['N'], params['omg'], params['tht'], 
                                                      params['nu'], params['kap'], time*params['T'], 
                                                      params['z']*params['L'] ) 
    base_flow_plots(  u , uz , bz , params['z'] , time , count , paths , params , plot_flag )
    #start_time_3 = datetime.now()
    A = fast_A( params , params['stat_mat'] , u/params['U'] , uz/params['omg'] , 
                bz/(params['N']**2. * np.sin(params['tht'])) , z , params['A0']*(0.+0.j) )
    #time_elapsed = datetime.now() - start_time_3
    #print('build A time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

  if case_flag == 'analytical_test':
    A = np.matrix([[-params['alpha'],-np.exp(1j*params['omg']*time)],[0.,-params['beta']]])
 
  if case_flag == 'Hills_equation':
    A = np.matrix([[0.,1.],[-params['a']-params['b']*np.cos(time),0.]],dtype=complex)
    # note that time must be in radians
    krk = np.matmul(A,Phin)

  if case_flag == 'inviscid_buoyancy':
    A21 = 2.*np.pi*1j*params['k']*np.sin(time)/(1.-params['c2']) #A21 = -2.*np.pi*np.sin(time)
    A22 = 1j*params['k']*np.cos(time)/(params['c2']-1.) #A22 = np.cos(time)
    A = np.matrix([[0.,1.],[A21,A22]],dtype=complex)

  if case_flag == 'forcing_test':
    #A = np.cos(time)
    omg = params['omg']
    A = np.cos(omg*time)

  if case_flag == 'forcing_test2':
    ustoke = stokes_solution( params, time, 0 )/params['U']
    #diag0 = np.zeros([int(params['Nz'])],dtype=complex)
    #for q in range(0,params['Nz']):
    #  diag0[q] = u[q]
    A = time*ustoke[3] # solution log(u) = -2pi*np.exp(-z[3]/dS)*cos(z[3]/dS)

  if case_flag == 'diffusion': # buoyancy equation diffusion (i.e. the heat equation)
    A = ( partial_zz( params['z'], params['Nz'], params['H'], 'neumann' , 'neumann' )[0] ) / (params['Pr']*params['Re'])
    # non-dimensional diffusion

  if case_flag == 'advection_diffusion': 
    u = ( rotating_solution( params, time, 0 )[1] ) / params['U'] # non-dimensional advection
    #print('max/min U = ',np.amax(abs(u)),np.amin(abs(u)))
    diag0 = np.zeros([int(params['Nz'])],dtype=complex)
    for q in range(0,params['Nz']):
      diag0[q] = u[q]*1j*params['k0'] - (params['k0']**2.+params['l0']**2.) / (params['Pr']*params['Re'])
    dzz = ( partial_zz( params['z'], params['Nz'], params['H'], 'neumann' , 'neumann' )[0] ) / (params['Pr']*params['Re'])
    A = diag0 + dzz

  if case_flag == 'zeta1':
    bz = ( rotating_solution( params, time, 1 )[3] ) / (params['N']**2.) 
    psi_inv = np.linalg.inv( partial_zz(  params['z'] , 'robin' , 'dirchlet' ) )
    dzz_zeta = partial_zz( params['z'] , 'open' , 'dirchlet' )
    dzz_b = partial_zz( params['z'] , 'neumann' , 'neumann' )
    eye_mat = np.eye( params['Nz'] , params['Nz'] , 0 , dtype=complex )
    A11 = ( dzz_zeta - (params['l0']**2.) * eye_mat ) / params['Re'] 
    A12 = params['Ri'] * 1j * params['l0'] * np.cos(params['tht']) * eye_mat 
    A21 = - 1j * params['l0'] * bz * psi_inv
    A22 = ( dzz_b - (params['k0']**2.) * eye_mat ) / ( params['Re'] * params['Pr'] )
    #A = np.matrix([[A11,A12],[A21,A22]],dtype=complex)
    Nz = int(params['Nz'])
    A = np.zeros([int(2*params['Nz']),int(2*params['Nz'])],dtype=complex)
    A[0:Nz,0:Nz] = A11
    A[0:Nz,int(Nz):int(2*Nz)] = A12
    A[Nz:int(2*Nz),0:Nz] = A21
    A[Nz:int(2*Nz),Nz:int(2*Nz)] = A22
   
  if case_flag == 'zeta2':
    # rotating solutions:
    u = ( rotating_solution( params, time, 0 )[1] ) / params['U'] # non-dimensional advection
    uzz = ( rotating_solution( params, time, 2 )[7] ) / params['U'] * params['L']**2. # non-dimensional advection
    bz = ( rotating_solution( params, time, 1 )[3] ) / (params['N']**2.) 
    #psi_inv = np.linalg.inv( partial_zz(  params['z'] , 'robin' , 'dirchlet' ) ) # non-dimensional
    #dzz_zeta = partial_zz( params['z'] , 'open' , 'dirchlet' ) # non-dimensional
    #dzz_b = partial_zz( params['z'] , 'neumann' , 'neumann' ) # non-dimensional
    #dz_b = partial_z( params['z'] , 'neumann' , 'neumann' ) # non-dimensional
    #eye_mat = np.eye( params['Nz'] , params['Nz'] , 0 , dtype=complex )
    A11 = -u*1j*params['a']*params['eye_matrix'] + ( params['dzz_zeta'] - (params['a']**2.) * params['eye_matrix'] ) / params['Re'] + np.dot(uzz*1j*params['a']*params['eye_matrix'],params['inv_psi'])
    A12 = params['Ri'] * ( params['dz_b'] * np.sin(params['tht']) - (1j*params['a']*np.cos(params['tht'])) * params['eye_matrix'] )   
    A21 = - bz * 1j * params['a'] * params['inv_psi']
    A22 = - u*1j*params['a']*params['eye_matrix'] + ( params['dzz_b'] - (params['a']**2.) * params['eye_matrix'] ) / ( params['Re'] * params['Pr'] )
    #A = np.matrix([[A11,A12],[A21,A22]],dtype=complex)  
    Nz = int(params['Nz'])
    A = np.zeros([int(2*params['Nz']),int(2*params['Nz'])],dtype=complex) 
    A[0:Nz,0:Nz] = A11
    A[0:Nz,int(Nz):int(2*Nz)] = A12
    A[Nz:int(2*Nz),0:Nz] = A21
    A[Nz:int(2*Nz),Nz:int(2*Nz)] = A22

  if case_flag == 'blennerhassett':
    #u = ( rotating_solution( params, time, 0 )[1] ) / params['U'] # non-dimensional advection
    #uzz = ( rotating_solution( params, time, 2 )[7] ) / params['U'] * params['L']**2. # non-dimensional advection
    #u = np.exp( -params['z']*params['Hd']/params['dS'] ) * np.cos( time - params['z']*params['Hd']/params['dS'] )
    #uzz = - 2.*(params['Hd']/params['dS'])**2. * np.exp( -params['z']*params['Hd']/params['dS'] ) * np.sin( time - params['z']*params['Hd']/params['dS'] )
    uS,uzS,uzzS = stokes_solution( params, time, 2 ) # dimensional solutions (input time is [0,2pi])
    uS = uS / params['U'] # non-dimensional
    uzzS = uzzS / (params['U']) * params['dS']**2. # non-dimensional

    #u = U * np.exp( -zd/dS ) * np.cos( time - zd/dS )
    #uz =  U/dS * np.exp( -zd/dS ) * ( np.sin( time - zd/dS ) - np.cos( time - zd/dS ) )
    #uzz = - 2.*U/(dS**2.) * np.exp( -zd/dS ) * np.sin( time - zd/dS ) 

    
    if params['plot_freq'] != 0:
        pfreq = params['plot_freq']
        #pfreq = params['plot_freq']*int(1./params['CFL'])

        if np.floor(count/pfreq) == count/pfreq:

            Phin0 = np.zeros([1,params['Nz']])
            for j00 in range(0,params['Nz']):
               Phin0[:,j00] = extrapolate_to_zero( np.real(Phin[:,j00]) , params['z'] , 8 )

            #print(np.shape(Phin))
            #print(np.shape(Phin[0,:]))
            #print(np.shape(params['z']))
            """
            H = params['H']
            plotname = params['phi_path'] +'%i.png' %(count)
            fig = plt.figure(figsize=(21,4.75))
            plt.subplot(141); 
            plt.plot(np.real(Phin[0,:]),params['z'],'b')
            plt.plot(Phin0[0,0],np.zeros([1,params['Nz']]),'ob',linewidth=3)
            plt.plot(np.real(Phin[1,:]),params['z'],'r')
            plt.plot(Phin0[0,1],np.zeros([1,params['Nz']]),'or',linewidth=3)
            plt.plot(np.real(Phin[2,:]),params['z'],'g')
            plt.plot(Phin0[0,2],np.zeros([1,params['Nz']]),'og',linewidth=3)
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            #plt.ylim([-H*0.05,H*1.05]); plt.grid()
            plt.ylim([-0.005,0.5]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            plt.subplot(142); 
            #plt.plot(np.real(Phin[0,:]),params['z'],'b')
            #plt.plot(Phin0,np.zeros([1,params['Nz']]),'or',linewidth=3)
            plt.plot(Phin0[0,0],np.zeros([1,params['Nz']]),'ob',linewidth=3)
            plt.plot(np.real(Phin[1,:]),params['z'],'r')
            plt.plot(Phin0[0,1],np.zeros([1,params['Nz']]),'or',linewidth=3)
            plt.plot(np.real(Phin[2,:]),params['z'],'g')
            plt.plot(Phin0[0,2],np.zeros([1,params['Nz']]),'og',linewidth=3)
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
            plt.axis([-0.02*np.amax(abs(np.real(Phin))),0.02*np.amax(abs(np.real(Phin))),-1.,1.1*H/25.]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            plt.subplot(143); 
            plt.semilogy(np.real(Phin),params['z'],'b')
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            plt.axis([-0.02*np.amax(abs(np.real(Phin))),0.02*np.amax(abs(np.real(Phin))),4e-2,1.]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            plt.subplot(144); 
            plt.semilogy(np.real(Phin[0,:]),params['z'],'b');
            #plt.semilogy(np.real(Phin[0,:]),params['z'],'r',linewidth=2)
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            plt.grid()
            plt.title(r"$|\zeta_0|$ = %.8f" %(np.amax(abs(Phin0))),fontsize=13)
            plt.savefig(plotname,format="png"); plt.close(fig);


            """
            mode = 0

            Psin = np.real(np.dot(params['inv_psi'],Phin))
            Psin0 = np.zeros([1,params['Nz']])
            for j00 in range(0,params['Nz']):
               Psin0[:,j00] = extrapolate_to_zero( Psin[:,j00] , params['z'] , 6 )

            Psi_mode = np.append(np.real(Psin0[:,mode]),np.real(Psin[:,mode]))
            z_mode = np.append(0.,params['z'])
            
            Psi_mode2 = np.append(np.real(Psin0[:,mode+1]),np.real(Psin[:,mode+1]))
            z_mode2 = np.append(0.,params['z'])

            Psi_plot = np.concatenate((np.real(Psin0),np.real(Psin)),axis=0)
            z_plot = z_mode


            H = params['H']
            plotname = params['psi_path'] +'%i.png' %(count)
            #fig = plt.figure(figsize=(21,4.75))
            fig = plt.figure(figsize=(10,4.75))
            plt.subplot(121); plt.plot(Psi_plot,z_plot,'b')
            plt.plot(Psi_mode2,z_mode2,color='cyan',linewidth=2)
            plt.plot(Psi_mode,z_mode,color='goldenrod',linewidth=2)
            plt.plot(Psin0,np.zeros([1,params['Nz']]),'or',linewidth=3)
            plt.xlabel(r"$\Psi$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            #plt.ylim([-H*0.05,H*1.05]); plt.grid()
            plt.ylim([-0.2,1.]); plt.grid()
            """
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            plt.subplot(142); plt.plot(Psi_plot,z_plot,'b')
            plt.plot(Psin0,np.zeros([1,params['Nz']]),'or',linewidth=3)
            plt.xlabel(r"$\Psi$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
            plt.axis([-0.02*np.amax(abs(np.real(Psin))),0.02*np.amax(abs(np.real(Psin))),-1.,1.1*H/25.]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            plt.subplot(143); plt.semilogy(Psi_plot,z_plot,'b')
            plt.xlabel(r"$\Psi$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            plt.axis([-0.02*np.amax(abs(np.real(Psin))),0.02*np.amax(abs(np.real(Psin))),4e-2,1.]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            """
            plt.subplot(122); plt.semilogy(Psi_plot,z_plot,'b')
            plt.semilogy(Psi_mode2,z_mode2,color='cyan',linewidth=2,linestyle='dashed')
            plt.semilogy(Psi_mode,z_mode,color='goldenrod',linewidth=2,linestyle='dashed')
            #plt.semilogy(Psin0,np.zeros([1,params['Nz']]),'or',linewidth=3)
            #plt.semilogy(np.real(Psin[params['Nz']-1,:]),params['z'],'r',linewidth=2)
            plt.xlabel(r"$\Psi$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            plt.grid()
            #plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            plt.title(r"$|\psi_0|$ = %.8f" %(np.amax(abs(Psin0))),fontsize=13)
            plt.savefig(plotname,format="png"); plt.close(fig);

            # extrapolation BC:

            Phin0 = np.zeros([1,params['Nz']])
            for j in range(0,params['Nz']):
                Phin0[0,j] = np.real( (extrapolate_to_zero( Phin[:,j] , params['z'] , 6 ))[0] )

            Phi_mode = np.append(np.real(Phin0[0,mode]),np.real(Phin[:,mode]))
            z_mode = np.append(0.,params['z'])

            Phi_mode2 = np.append(np.real(Phin0[0,mode+1]),np.real(Phin[:,mode+1]))
            z_mode2 = np.append(0.,params['z'])

            # Thom BC:

            #Phin0_Thom = np.zeros([1,params['Nz']])
            #Phin0_Thom[0,:] = np.real((np.matmul(params['inv_psi'],Phin))[0,:])*2./((params['z'][0])**2.) 
            Phin0_Thom = np.zeros([1,params['Nz']])
            Phin0_Thom[0,:] = np.real ( ((np.matmul(params['inv_psi'],Phin))[0,:])*3./((params['z'][0])**2.) - Phin[0,:]/2. ) # Woods (1954)

            Phi_mode = np.append(np.real(Phin0_Thom[0,mode]),np.real(Phin[:,mode]))
            z_mode = np.append(0.,params['z'])

            Phi_mode2 = np.append(np.real(Phin0_Thom[0,mode+1]),np.real(Phin[:,mode+1]))
            z_mode2 = np.append(0.,params['z'])

            Phi_plot = np.concatenate((np.real(Phin0_Thom),np.real(Phin)),axis=0)
            z_plot = z_mode
            
            H = params['H']
            plotname = params['phi_path'] +'%i.png' %(count)
            #fig = plt.figure(figsize=(21,4.75))
            fig = plt.figure(figsize=(10,4.75))
            plt.subplot(121); plt.plot(Phi_plot,z_plot,'b')
            plt.plot(Phi_mode2,z_mode2,color='cyan',linewidth=2)
            plt.plot(Phi_mode,z_mode,color='goldenrod',linewidth=2)
            #plt.plot(Phin0,np.zeros([1,params['Nz']]),'or',linewidth=3)
            plt.plot(Phin0_Thom,np.zeros([1,params['Nz']]),color='purple',linestyle='None',marker='o',linewidth=3)
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            #plt.ylim([-H*0.05,H*1.05]); plt.grid()
            plt.ylim([-0.2,1.]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            """
            plt.subplot(142); plt.plot(Phi_plot,z_plot,'b')
            plt.plot(Phin0,np.zeros([1,params['Nz']]),'or',linewidth=3)
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
            plt.axis([-0.02*np.amax(abs(np.real(Phin))),0.02*np.amax(abs(np.real(Phin))),-1.,1.1*H/25.]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            plt.subplot(143); plt.semilogy(np.real(Phin),params['z'],'b')
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            plt.axis([-0.02*np.amax(abs(np.real(Phin))),0.02*np.amax(abs(np.real(Phin))),4e-2,1.]); plt.grid()
            plt.title(r"t/T = %.4f, step = %i" %(time/params['T'],count),fontsize=13)
            """
            plt.subplot(122); plt.semilogy(Phi_plot,z_plot,'b');
            plt.semilogy(Phi_mode2,z_mode2,color='cyan',linewidth=2,linestyle='dashed')
            plt.semilogy(Phi_mode,z_mode,color='goldenrod',linewidth=2,linestyle='dashed')
            #plt.semilogy(np.real(Phin[:,mode]),((params['z'])[0])*np.ones(np.shape(params['z'])),color='red',linewidth=2,linestyle='dashed')
            #plt.semilogy(np.real(Phin[0,:]),params['z'],'r',linewidth=2)
            plt.xlabel(r"$\zeta$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
            plt.grid()
            plt.title(r"$|\zeta_0|$ = %.8f" %(np.amax(abs(Phin0))),fontsize=13)
            plt.savefig(plotname,format="png"); plt.close(fig);
   
  

    A = params['A0'] #np.zeros( [int(params['Nz']),int(params['Nz'])] , dtype=complex ) 
    if params['damper_scale'] == 0.: # no damper:
        A[:,:] = - uS*1j*params['a']*params['eye_matrix']*params['Re']/2. \
        + np.matmul(uzzS*1j*params['a']*params['eye_matrix']*params['Re']/2.,params['inv_psi']) \
        + ( params['dzz_zeta'] - (params['a']**2.*params['eye_matrix']) ) / 2. 
    elif params['damper_scale'] >= 1.: 
        A[:,:] = - uS*1j*params['a']*params['eye_matrix']*params['Re']/2. \
        + np.matmul(uzzS*1j*params['a']*params['eye_matrix']*params['Re']/2.,params['inv_psi']) \
        + ( params['dzz_zeta'] - (params['a']**2.*params['eye_matrix']) ) / 2. \
        - 5000. * params['eye_matrix'] * np.exp( (params['z']-params['H']) / ( params['H'] / params['damper_scale'] ) )
    else:
        print('ERROR: incorrect damper scale specified')


    # Phin = [ modes (one per IC) , z location ] rows = solutions? Needs to be columns. PROBLEM!!!!!
    krk = np.matmul(A,Phin) # Runge-Kutta coefficient (dot product of A operator and Phi)

    # get zeta_wall using the psi value from the first cell center, adjacent to the boundary (Thom 1933):
    
    """
    for j in range(0,params['Nz']): # for each mode:
        krk[j,0] = krk[j,0] + (params['lBC'] * zeta_wall[j]) # can I remove this from a loop? YES!
    """



    # NEED TO FIND A WAY TO SMOOTH THE FEILD SO THAT IT DOESN'T MESS UP THE TEMPORAL EVOLUTION?
    """
    zeta_wall = np.zeros([params['Nz']],dtype=complex)
    for j in range(0,params['Nz']):
        zeta_wall[j] = (extrapolate_to_zero( Phin[:,j] , params['z'] , 6 ))[0]
    krk[0,:] = krk[0,:] + (params['lBC'] * zeta_wall) / 2. # forward difference to the wall, for the first cell center   
    """
    """
    # Thom (1933) method:
    zeta_wall = ((np.matmul(params['inv_psi'],Phin))[0,:])*2./((params['z'][0])**2.) 
    # lBC comes from 'dirchlet' lower boundary condition (weight at -z[0])
    krk[0,:] = krk[0,:] + (params['lBC2'] * zeta_wall) # (divide by 2 from the vorticity equation)
    """
    # get wall vorticity:
    if params['grid_flag'] == 'cosine':
        zeta_wall = ((np.matmul(params['inv_psi'],Phin))[0,:])*2./((params['z'][0])**2.) # Thom (1933)
    else:
        #print('here!')
        zeta_wall = ((np.matmul(params['inv_psi'],Phin))[0,:])*3./((params['z'][0])**2.) - Phin[0,:]/2.  # Woods (1954)
    # now complete the finite difference stencil for the diffusion of vorticity at the first cell center:
    krk[0,:] = krk[0,:] + (params['lBC'] * zeta_wall) / 2. # the factor 2 is from the governing equations

  if case_flag == 'base_flow_test':
    b, u, v, bz, uz, vz, bzz, uzz, vzz = rotating_solution( params, time, 2 ) # dimensional
    base_flow = { 'u':(u/params['U']) , 'uz':(uz/params['U']*params['L']) ,
                  'uzz':(uzz/params['U']*params['L']**2.) , 'v':(v/params['U']) , 
                  'vz':(vz/params['U']*params['L']) , 'vzz':(vzz/params['U']*params['L']**2.) ,
                  'b':(b/(params['N']**2.*params['L'])) , 'bz':(bz/(params['N']**2.)) ,
                  'bzz':(bzz/(params['N']**2.)*params['L']) }
    u_path = '/home/bryan/git_repos/Floquet/base_flow/u/'
    uz_path = '/home/bryan/git_repos/Floquet/base_flow/uz/'
    uzz_path = '/home/bryan/git_repos/Floquet/base_flow/uzz/'
    v_path = '/home/bryan/git_repos/Floquet/base_flow/v/'
    vz_path = '/home/bryan/git_repos/Floquet/base_flow/vz/'
    vzz_path = '/home/bryan/git_repos/Floquet/base_flow/vzz/'
    b_path = '/home/bryan/git_repos/Floquet/base_flow/b/'
    bz_path = '/home/bryan/git_repos/Floquet/base_flow/bz/'
    bzz_path = '/home/bryan/git_repos/Floquet/base_flow/bzz/'
    paths = {'u_path':u_path, 'uz_path':uz_path, 'uzz_path':uzz_path, 
             'v_path':v_path, 'vz_path':vz_path, 'vzz_path':vzz_path, 
             'b_path':b_path, 'bz_path':bz_path, 'bzz_path':bzz_path}
    base_flow_plots(  base_flow , params , time , count, paths )
    krk = np.ones(np.shape(Phin)) 
 
  #if case_flag == 'base_flow_test':
  """
  else:
    # to use ATLAS BLAS library, both arguments in np.dot should be C-ordered. Check with:
    #print(Am.flags,Phi.flags)
    # Runge-Kutta coefficients
    #start_time_4 = datetime.now()
    krk = np.dot(A,Phin) 
    #time_elapsed = datetime.now() - start_time_4
    #print('A dot Phi time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed)) 
    check_matrix(krk,'krk')
  """
  return krk


def op_time_step( params , Phin , dt, stop_time, case_flag ):
  # uniform time step 4th-order Runge-Kutta time stepper

  time = 0. # non-dimensional time
  count = 0
  output_period = 10
  output_count = 0
   
  dt_mat = np.ones( np.shape(Phin) ) * dt
  while time < stop_time - dt: 

    if case_flag == 'boundary_layer':
      A = np.zeros([2,2])
    if case_flag == 'analytical_test':
      A = [[-params['alpha'],-np.exp(1j*params['omg']*time)],[0.,-params['beta']]]
    if case_flag == 'Hills_equation':
      A = np.matrix([[0.,1.],[-params['a']-params['b']*np.cos(time),0.]],dtype=complex)

    Phin = np.dot(scipy.linalg.expm(np.multiply(A,dt_mat)),Phin)

    #output_count = perturb_monitor( time , count , output_count , output_period , Phin , z , params , 'plot' )
    time = time + dt 
    count = count + 1
    check_matrix(Phin,'Phin')

  dtf =  stop_time - time
  if case_flag == 'boundary_layer':
    A = np.zeros([2,2])
  if case_flag == 'analytical_test':
    A = [[-params['alpha'],-np.exp(1j*params['omg']*time)],[0.,-params['beta']]]
  if case_flag == 'Hills_equation':
    A = np.matrix([[0.,1.],[-params['a']-params['b']*np.cos(time),0.]],dtype=complex)
  Phin = np.dot(scipy.linalg.expm(np.multiply(A, np.ones(np.shape(Phin))*dtf )),Phin)

  final_time = time + dtf # non-dimensional final time
  count = count + 1

  #print('ordered product method, final time = ', final_time)
  return Phin, final_time



#==============================================================================
# base flow functions (solutions)


def grid_choice( grid_flag , Nz , H ):
 # non-dimensional grid 
 if grid_flag == 'uniform': 
   z = np.linspace((H/Nz)/2. , H-(H/Nz)/2., num=Nz, endpoint=True) 

 if grid_flag == 'uniform 2': 
   z = np.linspace((H/(Nz-1))/2. , H-(H/(Nz-1))/2., num=int(Nz-1), endpoint=True)
   z1 = z[0]/2.
   z = np.append(z1,z) 
   #print(z)

 if grid_flag == 'hybrid tanh':
    H1 = H/25.
    Nz1 = int(Nz*13/16)
    #Nz1 = int(Nz*7/8)
    z1 = np.linspace((H1/Nz1)/2. , H1-(H1/Nz1)/2., num=Nz1, endpoint=True) 
    dz1 = H1-z1[Nz1-1]
    dz2 = 0.
    alpha = 1.1
    H2 = H-H1
    Nz2 = int(Nz-Nz1)
    z2 = np.zeros([Nz2])
    while dz2 <= dz1:
        gam = 1./5.
        z2 = H2*np.tanh( gam*np.linspace(Nz2/alpha, 0.125, num=int(Nz2))) / np.tanh(-gam*((Nz2+1)/alpha)) + np.ones([Nz2])*H2
        dz2 = z2[0]     
        alpha = alpha + 0.05 
    print('alpha =',alpha)
    z2 = z2 + H1  
    z = np.concatenate([z1,z2])
    if z[0] < 0.05:
         zplus = 0.05-z[0]
         z = z + zplus

 if grid_flag == 'hybrid cosine':
    # same as hybrid tanh, but use cosine so that there is clustering again at the top
    H1 = H/5.
    Nz1 = int(Nz*15/16)
    #Nz1 = int(Nz*3/4)
    print('Nz1 = ',Nz1)
    z1 = np.linspace((H1/Nz1)/2. , H1-(H1/Nz1)/2., num=Nz1, endpoint=True) 
    H2 = H-H1
    Nz2 = int(Nz-Nz1)
    #z2 = np.zeros([Nz2])
    z2 = -np.cos(((np.linspace(1., Nz2, num=Nz2))*2.-1.)/(2.*Nz2)*np.pi)*H2/2.+H2/2.
    z2 = z2 + H1  
    z = np.concatenate([z1,z2])

 if grid_flag == 'cosine': # half cosine grid
   z = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H
  
 if grid_flag == 'tanh':
   if H > 5.:
     #alpha = 1.5 # lower alpha for more points near bot boundary
     alpha = 1.5
     gam = 1./5.
     z = H*np.tanh( gam*np.linspace(Nz/alpha, 0.125, num=int(Nz))) / np.tanh(-gam*((Nz+1)/alpha)) + np.ones([Nz])*H #/ np.tanh( gam ) ( np.ones([Nz]) + 
     if z[0] < 0.05:
         zplus = 0.05-z[0]
         z = z + zplus

   else:
     print('Error: H to low for tanh grid')

 z = z[0:Nz] #/ 2. 
 dz = z[1:Nz]-z[0:Nz-1]
 #print(np.shape(z))
 return z,dz

"""
def nonrotating_solution( params, time ):  
 # FIX don't use, the u & b are 90 degrees out of phase
 
 # FIX all dimensional:  z and t
 z = params['z'] 
 U = params['U']
 N = params['N'] 
 omg = params['omg']
 Nz = params['Nz']
 Pr = params['Pr']
 nu = params['nu']
 kap = params['kap']
 
 tht = params['tht']
 thtcrit = params['thtc'] #ma.asin(omg/N) # radians 
 criticality = params['C']

 Bs = U*(N**2.0)*np.sin(tht)/omg 
 d1 = 0.; d2 = 0.; u1 = 0.; u2 = 0.; b1 = 0.; b2 = 0.; Ld = 0.

 if criticality > 1.0:
   d1 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) + \
       omg*(1.+Pr)/(4.*nu), -1./2.)
   d2 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) - \
       omg*(1.+Pr)/(4.*nu), -1./2.)
   Ld = np.power(((d1**2.+d2**2.)*(4.*(nu/Pr)**2. + \
      omg**2.*d1**2.*d2**2.))/(omg**2.*d1*d2), 1./4.) 
   u1 = 2.*kap/(d2*omg**2.*Ld**4.)+d2/(omg*Ld**4.) # s/m^3
   u2 = d1/(omg*Ld**4.)-2.*kap/(d1*omg**2.*Ld**4.) # s/m^3
   b1 = d1/(omg*Ld**4.) # s/m^3
   b2 = d2/(omg*Ld**4.) # s/m^3
   alpha1 = (omg*d1**2.-2.*nu/Pr)/(Ld*omg*d1)
   alpha2 = (2.*nu/Pr-omg*d2**2.)/(Ld*omg*d2)

 elif criticality < 1.0:
   d1 = np.power( omg*(1.+Pr)/(4.*nu) + \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
   d2 = np.power( omg*(1.+Pr)/(4.*nu) - \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
   Ld = ((d1-d2)*(2.*nu/Pr+omg*d1*d2))/(omg*d1*d2) # wall-normal buoyancy gradient lengthscale
   u1 = d2*(omg*d1**2.-2.*nu/Pr)/(Ld*omg*d1*d2) # unitless
   u2 = d1*(2.*nu/Pr-omg*d2**2.)/(Ld*omg*d1*d2) # unitless
   b1 = d1/Ld # unitless
   b2 = d2/Ld # unitless
   alpha1 = u1*(2.*kap*d1 + omg*d2**2.*d1)
   alpha2 = u1*(2.*kap*d2 - omg*d1**2.*d2)
   alpha3 = u2*( omg*d1**2.*d2 - 2.*kap*d2)
   alpha4 = u2*(2.*kap*d1 + omg*d2**2.*d1)
   beta1 = b1*(omg*d1**2.*d2-2.*kap*d2)
   beta2 = b1*(2.*kap*d1+omg*d1*d2**2.)
   beta3 = b2*(2.*kap*d1+omg*d1*d2**2.)
   beta4 = b2*(2.*kap*d2-omg*d1**2.*d2)
  
 elif criticality == 1.:
   print('ERROR: critical slope')
   return

 b = np.zeros([Nz])
 u = np.zeros([Nz])
 uz = np.zeros([Nz])
 bz = np.zeros([Nz])
 
 for i in range(0,Nz):
  
     if criticality < 1.:
       u[i] = - U*np.real( (u1*np.exp(-(1.+1j)*z[i]/d1) + \
                u2*np.exp(-(1.+1j)*z[i]/d2) - 1.)*np.exp(1j*omg*time) )
       uz[i] = U*np.real( (u1*(1.+1j)/d1*np.exp(-(1.+1j)*z[i]/d1) + \
               u2*(1.+1j)/d2*np.exp(-(1.+1j)*z[i]/d2) )*np.exp(1j*omg*time) )
       b[i] = - Bs*np.real( (b1*np.exp(-(1.0+1j)*z[i]/d1) - \
                b2*np.exp(-(1.+1j)*z[i]/d2) - 1.)*1j*np.exp(1j*omg*time) )
       bz[i] = - Bs*np.real( ( -(1.0+1j)/d1*b1*np.exp(-(1.0+1j)*z[i]/d1) + \
             (1.+1j)/d2*b2*np.exp(-(1.+1j)*z[i]/d2) )*1j*np.exp(1j*omg*time) )

     if criticality > 1.:
       u[i] = - U*np.real( (u1*(2.*kap*d1+omg*d1*d2**2.+ \
                1j*(2.*kap*d2-omg*d1**2.*d2))*np.exp((1j-1.0)*z[i]/d2)+ \
                u2*(omg*d1**2.*d2-2.*kap*d2+ \
                1j*(2.0*kap*d1+omg*d1*d2**2.))*np.exp(-(1j+1.)*z[i]/d1)- \
                1.)*np.exp(1j*omg*time) )
       uz[i] = U*np.real( ( (1j-1.0)/d2*( alpha1 + 1j*alpha2 )*np.exp((1j-1.0)*z[i]/d2) \
               -(1j+1.)/d1*(alpha3 + 1j*alpha4 )*np.exp(-(1j+1.)*z[i]/d1) )*np.exp(1j*omg*time) )
       b[i] = Bs*np.real( (b1*(omg*d1**2.*d2sp-2.*kap*d2+ \
                1j*(2.*kap*d1+omg*d1*d2**2.))*np.exp(-(1j+1.0)*z[i]/d1)+ \
                b2*(2.*kap*d1+omg*d1*d2**2.+ \
                1j*(2.*kap*d2-omg*d1**2.*d2))*np.exp((1j-1.0)*z[i]/d2)- \
                -1.)*1j*np.exp(1j*omg*time) )
       bz[i] = Bs*np.real( ( -(1j+1.0)/d1*( beta1 + 1j*beta2 )*np.exp(-(1j+1.0)*z[i]/d1)+ \
               (1j-1.0)/d2*( beta1 + 1j*beta2 )*np.exp((1j-1.0)*z[i]/d2) )*1j*np.exp(1j*omg*time) )

 if params['wall'] == 'moving':
   u = u - np.real(U*np.exp(1j*omg*time))
   b = b - np.real(Bs*1j*np.exp(1j*omg*time)) # check if this is correct!  #98BC37

 u = np.real(u) 
 b = np.real(b)
 uz = np.real(uz) 
 bz = np.real(bz)
 
 return  b, u, bz, uz
"""

def steady_rotating_solution( params, order ):
 # Wunsch (1970) rotating, steady solution

 # all dimensional: 
 z = params['z']*params['dS'] # m, dimensional grid 
 N = params['N'] 
 Nz = params['Nz']
 Pr = params['Pr']
 nu = params['nu']
 kap = params['kap']
 f = params['f'] 
 tht = params['tht']

 Pr = nu / kap
 cot = np.cos(tht) / np.sin(tht)

 d0 = ( ( f*np.cos(tht) / (2.*nu) )**2. + ( N*np.sin(tht)/2. )**2./(nu*kap) )**(-1./4.)
 vg = -Pr*d0*f*cot # geostrophic far field velocity

 Z = z/d0
 b0 = N**2.*d0*np.cos(tht)*np.exp(-Z)*np.cos(Z)
 u0 = 2.*kap*cot/d0*np.exp(-Z)*np.sin(Z)
 v0 = Pr*f*d0*cot*np.exp(-Z)*np.cos(Z) + vg

 bz0 = -N**2.*np.cos(tht)*np.exp(-Z)*(np.cos(Z)+np.sin(Z))
 uz0 = 2.*kap*cot*(d0**(-2.))*np.exp(-Z)*(np.cos(Z)-np.sin(Z))
 vz0 = -Pr*f*cot*np.exp(-Z)*(np.cos(Z)+np.sin(Z)) 

 if order < 1:
   return np.real(b0), np.real(u0), np.real(v0)
 if order == 1:
   return np.real(b0), np.real(u0), np.real(v0), np.real(bz0), np.real(uz0), np.real(vz0)


def blank_stokes_solution( params, time, order ):

 u = np.zeros([params['Nz']]); 
 if order >= 1:
   uz = np.zeros([params['Nz']]); 
 if order == 2:
   uzz = np.zeros([params['Nz']]); 

 if order < 1:
   return np.real(u)
 if order == 1:
   return np.real(u), np.real(uz)
 if order == 2:
   return np.real(u), np.real(uz), np.real(uzz)


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


def blank_rotating_solution( params, time, order ):

 b = np.zeros([params['Nz']]); u = np.zeros([params['Nz']]); v = np.zeros([params['Nz']]); 
 if order >= 1:
   bz = np.zeros([params['Nz']]); uz = np.zeros([params['Nz']]); vz = np.zeros([params['Nz']]);
 if order == 2:
   bzz = np.zeros([params['Nz']]); uzz = np.zeros([params['Nz']]); vzz = np.zeros([params['Nz']]);

 # dimensional output:
 if order < 1:
   return np.real(b), np.real(u), np.real(v)
 if order == 1:
   return np.real(b), np.real(u), np.real(v), np.real(bz), np.real(uz), np.real(vz)
 if order == 2:
   return np.real(b), np.real(u), np.real(v), np.real(bz), np.real(uz), np.real(vz), np.real(bzz), np.real(uzz), np.real(vzz)



 
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


#==============================================================================
# z derivative functions


def weights2( z0 , z1 , z2 , z3 , zj ):
 # Lagrange polynomial weights for second derivative
 l0 = 1./(z0-z1) * ( 1./(z0-z2) * (zj-z3)/(z0-z3) + 1./(z0-z3) * (zj-z2)/(z0-z2) ) + \
      1./(z0-z2) * ( 1./(z0-z1) * (zj-z3)/(z0-z3) + 1./(z0-z3) * (zj-z1)/(z0-z1) ) + \
      1./(z0-z3) * ( 1./(z0-z1) * (zj-z2)/(z0-z2) + 1./(z0-z2) * (zj-z1)/(z0-z1) )
 l1 = 1./(z1-z0) * ( 1./(z1-z2) * (zj-z3)/(z1-z3) + 1./(z1-z3) * (zj-z2)/(z1-z2) ) + \
      1./(z1-z2) * ( 1./(z1-z0) * (zj-z3)/(z1-z3) + 1./(z1-z3) * (zj-z0)/(z1-z0) ) + \
      1./(z1-z3) * ( 1./(z1-z0) * (zj-z2)/(z1-z2) + 1./(z1-z2) * (zj-z0)/(z1-z0) )
 l2 = 1./(z2-z0) * ( 1./(z2-z1) * (zj-z3)/(z2-z3) + 1./(z2-z3) * (zj-z1)/(z2-z1) ) + \
      1./(z2-z1) * ( 1./(z2-z0) * (zj-z3)/(z2-z3) + 1./(z2-z3) * (zj-z0)/(z2-z0) ) + \
      1./(z2-z3) * ( 1./(z2-z0) * (zj-z1)/(z2-z1) + 1./(z2-z1) * (zj-z0)/(z2-z0) )
 l3 = 1./(z3-z0) * ( 1./(z3-z1) * (zj-z2)/(z3-z2) + 1./(z3-z2) * (zj-z1)/(z3-z1) ) + \
      1./(z3-z1) * ( 1./(z3-z0) * (zj-z2)/(z3-z2) + 1./(z3-z2) * (zj-z0)/(z3-z0) ) + \
      1./(z3-z2) * ( 1./(z3-z0) * (zj-z1)/(z3-z1) + 1./(z3-z1) * (zj-z0)/(z3-z0) )
 return l0, l1, l2, l3

"""
def partial_zz_old( z, Nz, H, lower_BC_flag , upper_BC_flag ):
 # second derivative, permiting non-uniform grids
 #Nz = params['Nz']
 #z = params['z']
 #H = params['H']

 # 2nd order accurate (truncated 3rd order terms), variable grid
 diagm1 = np.zeros([Nz-1],dtype=complex)
 diag0 = np.zeros([Nz],dtype=complex)
 diagp1 = np.zeros([Nz-1],dtype=complex)
 for j in range(1,Nz-1):
     denom = 1./2. * ( z[j+1] - z[j-1] ) * ( z[j+1] - z[j] ) * ( z[j] - z[j-1] )  
     diagm1[j-1] = ( z[j+1] - z[j] ) / denom
     diagp1[j] =   ( z[j] - z[j-1] ) / denom
     diag0[j] =  - ( z[j+1] - z[j-1] ) / denom
 pzz = np.diag(diagp1,k=1) + np.diag(diagm1,k=-1) + np.diag(diag0,k=0) 

 # lower (wall) BC sets variable to zero at the wall
 zj = z[0] # location of derivative for lower BC (first cell center)
 l0, l1, l2, l3 = weights2( -z[0] , z[0] , z[1] , z[2] , zj )
 if lower_BC_flag == 'dirchlet':
   l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
 if lower_BC_flag == 'neumann':
   l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
 pzz[0,0:3] = [ l1 , l2 , l3 ]
 lBC = l0

 # upper (far field) BC
 zj = z[Nz-1] # location of derivative for upper BC
 l0, l1, l2, l3 = weights2( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , zj )
 if upper_BC_flag == 'dirchlet':
   l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
 pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 return pzz,lBC
"""

def partial_zz( params , lower_BC_flag , upper_BC_flag ):
 # second derivative, permiting non-uniform grids
 z = params['z']*params['Hd']
 Nz = params['Nz']
 H = params['Hd']
 wall_flag = params['wall_flag']

 # 2nd order accurate (truncated 3rd order terms), variable grid
 diagm1 = np.zeros([Nz-1])
 diag0 = np.zeros([Nz])
 diagp1 = np.zeros([Nz-1])
 for j in range(1,Nz-1):
     denom = 1./2. * ( z[j+1] - z[j-1] ) * ( z[j+1] - z[j] ) * ( z[j] - z[j-1] )  
     diagm1[j-1] = ( z[j+1] - z[j] ) / denom
     diagp1[j] =   ( z[j] - z[j-1] ) / denom
     diag0[j] =  - ( z[j+1] - z[j-1] ) / denom
 pzz = np.diag(diagp1,k=1) + np.diag(diagm1,k=-1) + np.diag(diag0,k=0) 

 # lower (wall) BC sets variable to zero at the wall
 zj = z[0] # location of derivative for lower BC (first cell center)
 if lower_BC_flag == 'dirchlet':
   l0, l1, l2, l3 = weights2( -z[0] , z[0] , z[1] , z[2] , zj ); lBC2 = l0
   pzz[0,0:3] = [ l1 - l0 , l2 , l3 ] # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
 if lower_BC_flag == 'neumann':
   l0, l1, l2, l3 = weights2( -z[0] , z[0] , z[1] , z[2] , zj ); lBC2 = l0 
   pzz[0,0:3] = [ l1 + l0 , l2 , l3 ] # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
 if lower_BC_flag == 'open':
   l0, l1, l2, l3 = weights2( z[0] , z[1] , z[2] , z[3] , zj )
   pzz[0,0:4] = [ l0 , l1 , l2 , l3 ]
 if lower_BC_flag == 'robin':
   l0, l1, l2, l3 = weights2( -z[1] , -z[0] , z[0] , z[1] , zj ) 
   pzz[0,0:2] = [ l1 + l2 , l3 - l0 ] # combined Neumann and Dirchlet at z = 0
 #if lower_BC_flag == 'thom':
 #  l0, l1, l2, l3 = weights2( -z[1] , -z[0] , z[0] , z[1] , zj ) 
 #  pzz[0,0:2] = [ l1 + l2 , l3 - l0 ] # combined Neumann and Dirchlet at z = 0
   
 # upper (far field) BC
 zj = z[Nz-1] # location of derivative for upper BC
 if upper_BC_flag == 'dirchlet':
   l0, l1, l2, l3 = weights2( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , zj )
   pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 - l3 ] # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l0, l1, l2, l3 = weights2( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , zj ) 
   pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 + l3 ] # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
 if upper_BC_flag == 'open':
   l0, l1, l2, l3 = weights2( z[Nz-4] , z[Nz-3] , z[Nz-2] , z[Nz-1] , zj ) 
   pzz[Nz-1,Nz-4:Nz] = [ l0 , l1 , l2 , l3 ]

 # dimensional output:
 if wall_flag == 'moving':
   #print('moving wall')
   return pzz,lBC2
 else:
   return pzz


def partial_z( params , lower_BC_flag , upper_BC_flag ):
 # first-order derivative matrix 
 z = params['z']*params['Hd']
 Nz = params['Nz']
 H = params['Hd']
 wall_flag = params['wall_BC_flag']
 #print(H,np.amax(z),np.amin(z))

 # interior points, variable grid
 diagm1 = np.zeros([Nz-1])
 diag0 = np.zeros([Nz])
 diagp1 = np.zeros([Nz-1])
 for j in range(1,Nz-1):
   denom = ( ( z[j+1] - z[j] ) * ( z[j] - z[j-1] ) * ( ( z[j+1] - z[j] ) + ( z[j] - z[j-1] ) ) ) 
   diagm1[j-1] = - ( z[j+1] - z[j] )**2. / denom
   diagp1[j] =   ( z[j] - z[j-1] )**2. / denom
   diag0[j] =  ( ( z[j+1] - z[j] )**2. - ( z[j] - z[j-1] )**2. ) / denom
 pz = np.diag(diagp1,k=1) + np.diag(diagm1,k=-1) + np.diag(diag0,k=0) 

 # lower BC
 l0, l1, l2, l3 = weights( -z[0] , z[0] , z[1] , z[2] , z[0] ); lBC = l0
 if lower_BC_flag == 'dirchlet':
   l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
 if lower_BC_flag == 'neumann':
   l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
 pz[0,0:3] = [ l1 , l2 , l3 ]
   
 # upper (far field) BC
 l0, l1, l2, l3 = weights( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , z[Nz-1] )
 if upper_BC_flag == 'dirchlet':
   l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
 pz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 
 # dimensional output:
 if wall_flag == 'moving':
   return pz,lBC
 else:
   return pz


def weights( z0 , z1 , z2 , z3 , zj ):
 # Lagrange polynomial weights for first derivative
 l0 = 1./(z0-z1) * (zj-z2)/(z0-z2) * (zj-z3)/(z0-z3) + \
      1./(z0-z2) * (zj-z1)/(z0-z1) * (zj-z3)/(z0-z3) + \
      1./(z0-z3) * (zj-z1)/(z0-z1) * (zj-z2)/(z0-z2)
 l1 = 1./(z1-z0) * (zj-z2)/(z1-z2) * (zj-z3)/(z1-z3) + \
      1./(z1-z2) * (zj-z0)/(z1-z0) * (zj-z3)/(z1-z3) + \
      1./(z1-z3) * (zj-z0)/(z1-z0) * (zj-z2)/(z1-z2)
 l2 = 1./(z2-z0) * (zj-z1)/(z2-z1) * (zj-z3)/(z2-z3) + \
      1./(z2-z1) * (zj-z0)/(z2-z0) * (zj-z3)/(z2-z3) + \
      1./(z2-z3) * (zj-z0)/(z2-z0) * (zj-z1)/(z2-z1)
 l3 = 1./(z3-z0) * (zj-z1)/(z3-z1) * (zj-z2)/(z3-z2) + \
      1./(z3-z1) * (zj-z0)/(z3-z0) * (zj-z2)/(z3-z2) + \
      1./(z3-z2) * (zj-z0)/(z3-z0) * (zj-z1)/(z3-z1)
 return l0, l1, l2, l3


def fornberg_weights(z,x,m):
# From Bengt Fornbergs (1998) SIAM Review paper.
#  	Input Parameters
#	z location where approximations are to be accurate,
#	x(0:nd) grid point locations, found in x(0:n)
#	n one less than total number of grid points; n must
#	not exceed the parameter nd below,
#	nd dimension of x- and c-arrays in calling program
#	x(0:nd) and c(0:nd,0:m), respectively,
#	m highest derivative for which weights are sought,
#	Output Parameter
#	c(0:nd,0:m) weights at grid locations x(0:n) for derivatives
#	of order 0:m, found in c(0:n,0:m)
#      	dimension x(0:nd),c(0:nd,0:m)

  n = np.shape(x)[0]-1
  c = np.zeros([n+1,m+1])
  c1 = 1.0
  c4 = x[0]-z
  for k in range(0,m+1):  
    for j in range(0,n+1): 
      c[j,k] = 0.0
  c[0,0] = 1.0
  for i in range(0,n+1):
    mn = min(i,m)
    c2 = 1.0
    c5 = c4
    c4 = x[i]-z
    for j in range(0,i):
      c3 = x[i]-x[j]
      c2 = c2*c3
      if (j == i-1):
        for k in range(mn,0,-1): 
          c[i,k] = c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2
      c[i,0] = -c1*c5*c[i-1,0]/c2
      for k in range(mn,0,-1):
        c[j,k] = (c4*c[j,k]-k*c[j,k-1])/c3
      c[j,0] = c4*c[j,0]/c3
    c1 = c2
  return c


def diff_matrix( params , lower_BC_flag , upper_BC_flag , diff_order , stencil_size ):
 # uses ghost nodes for dirchlet/neumann/thom bcs.
 # make stencil size odd!
 # no interpolation
 z = params['z'] # non-dimensional
 Nz = params['Nz']
 H = params['H'] #params['grid_scale']
 wall_BC_flag = params['wall_BC_flag']
 
 if stencil_size == 3:
   Dm1 = np.zeros([Nz-1])
   D0 = np.zeros([Nz])
   Dp1 = np.zeros([Nz-1])
   for j in range(1,Nz-1):
     Dm1[j-1],D0[j],Dp1[j] = fornberg_weights(z[j],z[j-1:j+2],diff_order)[:,diff_order]
   pzz = np.diag(Dp1,k=1) + np.diag(Dm1,k=-1) + np.diag(D0,k=0) 

   # lower (wall) BC sets variable to zero at the wall
   if lower_BC_flag == 'dirchlet 2':
     #lw, l0, l1 = fornberg_weights(z[0], np.append(0.,z[0:2]) ,diff_order)[:,diff_order]
     #lBC = lw
     #pzz[0,0:2] = [ l0 , l1 ]
     #lw, l0, l1, l2 = fornberg_weights(z[0], np.append(0.,z[0:3]) ,diff_order)[:,diff_order]
     #lBC = lw
     #pzz[0,0:3] = [ l0 , l1 , l2 ]
     #lw, l0, l1, l2, l3, l4  = fornberg_weights(z[0], np.append(0.,z[0:5]) ,diff_order)[:,diff_order]
     #lBC = lw
     #pzz[0,0:5] = [ l0 , l1 , l2 , l3 , l4 ]
     lw, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9  = fornberg_weights(z[0], np.append(0.,z[0:10]) ,diff_order)[:,diff_order]
     lBC = lw
     pzz[0,0:10] = [ l0, l1, l2, l3, l4, l5, l6, l7, l8, l9 ]
     #lw, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15  = fornberg_weights(z[0], np.append(0.,z[0:16]) ,diff_order)[:,diff_order]
     #lBC = lw
     #pzz[0,0:16] = [ l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15 ]

   if lower_BC_flag == 'dirchlet':
     l0, l1, l2, l3 = fornberg_weights(z[0], np.append(-z[0],z[0:3]) ,diff_order)[:,diff_order]
     lBC = l0
     #print(lBC)
     l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
     pzz[0,0:3] = [ l1 , l2 , l3 ]
     """
     lm1, lm0, l0, l1, l2 = fornberg_weights(z[0], np.append(-z[1],np.append(-z[0],z[0:3])) ,diff_order)[:,diff_order]
     l0 = l0 - lm0
     l1 = l1 - lm1
     lBC1 = lm0
     lBC2 = lm1
     pzz[0,0:3] = [ l0 , l1 , l2 ]
     """
     """
     l0, l1 = fornberg_weights(z[0], np.append(-z[0],z[0:1]) ,diff_order)[:,diff_order]
     lBC = l0
     #print(lBC)
     l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
     pzz[0,0] = l1
     """

   if lower_BC_flag == 'thom 2':
     # do not use for zeta!
     # uniform grid, second derivative, psi BC only
     # five point stencil, centered at the first cell center, uniform grid
     """
     dz = 2.*z[0]
     lm2 = -1./(12.*dz**2.)
     lm1 = 4./(3.*dz**2.)
     l0 = -5./(2.*dz**2.)
     l1 = 4./(3.*dz**2.)
     l2 = -1./(12.*dz**2.)
     """
     """
     l0, l1, l2, l3, l4 = fornberg_weights(z[0], np.append(-z[0],z[0:4]) ,diff_order)[:,diff_order]
     #l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
     pzz[0,0:4] = np.array([ 0. , l2 , l3 , l4 ])
     """
     l0, l1, l2, l3 = fornberg_weights(z[0], np.append(-z[1],np.append(-z[0],z[0:2])), diff_order)[:, diff_order]
     l2 = l2 - l1
     l3 = l3 + l0
     pzz[0,0:2] = np.array([ l2 , l3 ])    
     # try with high order inside?

   if lower_BC_flag == 'neumann':
     #print('here!')
     # is their a problem here?
     #l0, l1, l2, l3 = fornberg_weights(z[0], np.append(-z[0],z[0:3]) ,diff_order)[:,diff_order]

     l0, l1, l2, l3 = fornberg_weights(z[0], np.append(-z[0],z[0:3]) ,diff_order)[:,diff_order]
     #l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
     pzz[0,0:3] = np.array([ l1 + l0 , l2 , l3 ])
     """
     l0, l1, l2, l3, l4 = fornberg_weights(z[0], np.append(-z[0],z[0:4]) ,diff_order)[:,diff_order]
     #l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
     pzz[0,0:4] = np.array([ l1 + l0 , l2 , l3 , l4 ]) 
     """

     """
     l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15 = fornberg_weights(z[0], np.append(-z[0],z[0:15]) ,diff_order)[:,diff_order]
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
     pzz[0,0:15] = [ l1 , l2 , l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15 ]
     """

     """
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[0], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
     pzz[0,0:6] = [ l1 , l2 , l3, l4, l5, l6 ]
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[1], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     pzz[1,0:6] = np.array([ l1 + l0 , l2 , l3, l4, l5, l6  ])
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[2], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     pzz[2,0:6] = np.array([ l1 + l0 , l2 , l3, l4, l5, l6  ])
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[3], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     pzz[3,0:6] = np.array([ l1 + l0 , l2 , l3, l4, l5, l6  ])
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[4], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     pzz[4,0:6] = np.array([ l1 + l0 , l2 , l3, l4, l5, l6  ])
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[5], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     pzz[5,0:6] = np.array([ l1 + l0 , l2 , l3, l4, l5, l6  ])
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[6], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     pzz[6,0:6] = np.array([ l1 + l0 , l2 , l3, l4, l5, l6  ])
     """

     """     
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[0], np.append(-z[0],z[0:6]) ,diff_order)[:,diff_order]
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
     pzz[0,0:6] = [ l1 , l2 , l3, l4, l5, l6 ]

     # to do the Neumann BC, look for the derivative at z[0] = first center, which must be equal to -z[0]
     l0, l1, l2 = fornberg_weights(z[0], z[0:3] ,diff_order)[:,diff_order]
     l0 = 2.*l0
     pzz[0,0:3] = [ l0 , l1 , l2 ]
     """

   if lower_BC_flag == 'thom':
     """
     l0, l1, l2, l3, l4, l5 = fornberg_weights(z[0], np.append(-z[1],np.append(-z[0],np.append(0.,z[0:3]))), diff_order)[:, diff_order]
     #l2 = 0.
     l3 = l3 + l1
     #l4 = l4 + l0
     pzz[0,0:3] = [l3, l4, l5]
     
     l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12 = fornberg_weights(z[0], np.append(-z[1],np.append(-z[0],np.append(0.,z[0:10]))), diff_order)[:, diff_order]
     #l2 = 0.
     l3 = l3 + l1
     #l4 = l4 + l0
     pzz[0,0:10] = [l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]
     
     l0, l1, l2, l3, l4 = fornberg_weights(z[0], np.append(-z[0],np.append(0.,z[0:3])), diff_order)[:, diff_order]
     #l1 = 0. # true but unecessary to specify
     l2 = l2 + l0
     pzz[0,0:3] = [l2, l3, l4]
     """

     l0, l1, l2, l3 = fornberg_weights(z[0], z[0:4] ,diff_order)[:,diff_order]
     pzz[0,0:4] = np.array([1.,0.,0.,0.])

   if lower_BC_flag == 'open':
     l0, l1, l2, l3, l4 = fornberg_weights(z[0], z[0:5], diff_order)[:, diff_order]
     pzz[0,0:5] = [l0, l1, l2, l3, l4]


   # upper (far field) BC 
   if upper_BC_flag == 'dirchlet':
     #l0, l1, l2, l3, l4, l5 = fornberg_weights(z[Nz-1], np.append(z[Nz-5:Nz],H) ,diff_order)[:,diff_order]
     #pzz[Nz-1,Nz-5:Nz] = [ l0, l1, l2, l3, l4 ] # effectively sets l5, the variable at the boundary, to zero, without zeroing out the gradient.

     l0, l1, l2, l3 = fornberg_weights(z[Nz-1], np.append(z[Nz-3:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
     l2 = l2 - l3 # at z=H (sets phi_ghost = - phi_N, cell centers)
     pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
     """
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[Nz-1], np.append(z[Nz-6:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
     l5 = l5 - l6 # at z=H (sets phi_ghost = - phi_N, cell centers)
     #l5 = l5 - l6 # at z=H (sets phi_ghost = - phi_N, cell centers)
     pzz[Nz-1,Nz-6:Nz] = [ l0 , l1 , l2 , l3 , l4 , l5 ]
     """     
   if upper_BC_flag == 'neumann':
     l0, l1, l2, l3 = fornberg_weights(z[Nz-1], np.append(z[Nz-3:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
     l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
     pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
   if upper_BC_flag == 'open':
     l0, l1, l2, l3, l4 = fornberg_weights(z[Nz-1], z[Nz-5:Nz], diff_order)[:, diff_order]
     pzz[Nz-1,Nz-5:Nz] = [l0, l1, l2, l3, l4]
   if upper_BC_flag == 'thom':
     l0, l1, l2, l3, l4, l5, l6 = fornberg_weights(z[Nz-1], np.append(z[Nz-3:Nz],np.append(H,np.append(np.append(H + (H-z[Nz-1]),H+(H-z[Nz-2])),H+(H-z[Nz-3])))) ,diff_order)[:,diff_order]
     l3 = 0.
     l2 = l2 + l4 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
     l1 = l1 + l5
     l0 = l0 + l6
     pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]


 """
 if stencil_size == 5:
   Dm2 = np.zeros([Nz-2])
   Dm1 = np.zeros([Nz-1])
   D0 = np.zeros([Nz])
   Dp1 = np.zeros([Nz-1])
   Dp2 = np.zeros([Nz-2])
   for j in range(2,Nz-2):
     Dm2[j-2],Dm1[j-1],D0[j],Dp1[j],Dp2[j] = fornberg_weights(z[j],z[j-2:j+3],diff_order)[:,diff_order]
   pzz = np.diag(Dp2,k=2) + np.diag(Dp1,k=1) + np.diag(D0,k=0) + np.diag(Dm1,k=-1) + np.diag(Dm2,k=-2) 
 

   # lower (wall) BC sets variable to zero at the wall
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[0], np.append(-z[0],z[0:5]) ,diff_order)[:,diff_order]
   if lower_BC_flag == 'dirchlet':
     l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
   if lower_BC_flag == 'neumann':
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
   pzz[0,0:5] = [ l1 , l2 , l3, l4 , l5 ]
   # lower (wall) BC sets variable to zero at the wall
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[1], np.append(-z[0],z[0:5]) ,diff_order)[:,diff_order]
   if lower_BC_flag == 'dirchlet':
     l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
   if lower_BC_flag == 'neumann':
     l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
   pzz[1,0:5] = [ l1 , l2 , l3, l4 , l5 ]
   # upper (far field) BC
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[Nz-1], np.append(z[Nz-5:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
   if upper_BC_flag == 'dirchlet':
     l4 = l4 - l5 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
   if upper_BC_flag == 'neumann':
     l4 = l4 + l5 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
   pzz[Nz-1,Nz-5:Nz] = [ l0 , l1 , l2 , l3 , l4 ]
   # upper (far field) BC
   l0, l1, l2, l3, l4, l5 = fornberg_weights(z[Nz-2], np.append(z[Nz-5:Nz],H + (H-z[Nz-1])) ,diff_order)[:,diff_order]
   if upper_BC_flag == 'dirchlet':
     l4 = l4 - l5 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
   if upper_BC_flag == 'neumann':
     l4 = l4 + l5 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
   pzz[Nz-2,Nz-5:Nz] = [ l0 , l1 , l2 , l3 , l4 ]
 """
 #print(lBC)
 #pzz = np.multiply(pzz,np.ones(np.shape(pzz)),dtype=complex) # make pzz complex
 if wall_BC_flag == 'BC':
   return pzz,lBC
 else:
   return pzz

def extrapolate_to_zero( data , z , Npoints ):
    #print(np.shape(z[0:Npoints]))
    #print(np.shape(np.transpose(fornberg_weights(0.,(z[0:Npoints]),0))))
    #print(np.shape(data[0:Npoints]))
    data0 = np.dot(np.transpose(fornberg_weights(0.,(z[0:Npoints]),0)),data[0:Npoints])
    return data0





#==============================================================================
# error related functions

def check_matrix(self,string):
 if np.any(np.isnan(self)) == True:
  print('NaN detected in '+ string)
 if np.any(np.isinf(self)) == True:
  print('Inf detected in '+ string)
 return


#==============================================================================
# plotting functions

def base_flow_plots(  base_flow , params , time , count , paths ):
  freq = 1 #np.floor( (1./params['dt'])/200. )
  #print(np.shape(base_flow))
  #print('base_flow count: ',count)

  #u = base_flow['u']
  if np.floor(count/params['freq']) == count/params['freq']:
 
     yaxis1 = [-0.05,params['H']]
     yaxis2 = [-0.05,1.]
     yaxis3 = [0.,0.03]

     plotname = paths['u_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.5))
     plt.subplot(131); plt.plot(base_flow['u'],params['z'],'b')
     plt.xlabel(r"$u/U_\infty$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis1); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132); plt.plot(base_flow['u'],params['z'],'b')
     plt.xlabel(r"$u/U_\infty$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
     plt.ylim(yaxis2); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133); plt.semilogy(base_flow['u'],params['z'],'b')
     plt.xlabel(r"$u/U_\infty$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis3); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

     plotname = paths['uz_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.5))
     plt.subplot(131); plt.plot(base_flow['uz'],params['z'],'b')
     plt.xlabel(r"$u_z/\omega$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis1); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132); plt.plot(base_flow['uz'],params['z'],'b')
     plt.xlabel(r"$u_z/\omega$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
     plt.ylim(yaxis2); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133); plt.semilogy(base_flow['uz'],params['z'],'b')
     plt.xlabel(r"$u_z/\omega$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis3); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

     plotname = paths['uzz_path'] +'%i.png' %(count) # ( m/s 1/m^2 ) / ( 1/s * 1/m)  omg/L = omg**2/U  (1/s^2 * s/m) = 1/(ms)
     fig = plt.figure(figsize=(16,4.5))
     plt.subplot(131); plt.plot(base_flow['uzz'],params['z'],'b')
     plt.xlabel(r"$u_{zz}L\omega^{-1}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis1); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132); plt.plot(base_flow['uzz'],params['z'],'b')
     plt.xlabel(r"$u_{zz}L\omega^{-1}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
     plt.ylim(yaxis2); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133); plt.semilogy(base_flow['uzz'],params['z'],'b')
     plt.xlabel(r"$u_{zz}L\omega^{-1}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis3); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

     
     if params['Ro'] < np.inf:
       print(params['Ro'])  
       plotname = paths['v_path'] +'%i.png' %(count)
       fig = plt.figure(figsize=(16,4.5))
       plt.subplot(131); plt.plot(base_flow['v'],params['z'],'b')
       plt.xlabel(r"$v/U_\infty$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
       plt.ylim(yaxis1); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.subplot(132); plt.plot(base_flow['v'],params['z'],'b')
       plt.xlabel(r"$v/U_\infty$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
       plt.ylim(yaxis2); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.subplot(133); plt.semilogy(base_flow['v'],params['z'],'b')
       plt.xlabel(r"$v/U_\infty$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
       plt.ylim(yaxis3); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.savefig(plotname,format="png"); plt.close(fig);

       plotname = paths['vz_path'] +'%i.png' %(count)
       fig = plt.figure(figsize=(16,4.5))
       plt.subplot(131); plt.plot(base_flow['vz'],params['z'],'b')
       plt.xlabel(r"$v_z/\omega$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
       plt.ylim(yaxis1); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.subplot(132); plt.plot(base_flow['vz'],params['z'],'b')
       plt.xlabel(r"$v_z/\omega$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
       plt.ylim(yaxis2); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.subplot(133); plt.semilogy(base_flow['vz'],params['z'],'b')
       plt.xlabel(r"$v_z/\omega$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
       plt.ylim(yaxis3); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.savefig(plotname,format="png"); plt.close(fig);

       plotname = paths['vzz_path'] +'%i.png' %(count) # ( m/s 1/m^2 ) / ( 1/s * 1/m)  omg/L = omg**2/U  (1/s^2 * s/m) = 1/(ms)
       fig = plt.figure(figsize=(16,4.5))
       plt.subplot(131); plt.plot(base_flow['vzz'],params['z'],'b')
       plt.xlabel(r"$v_{zz}L\omega^{-1}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
       plt.ylim(yaxis1); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.subplot(132); plt.plot(base_flow['vzz'],params['z'],'b')
       plt.xlabel(r"$v_{zz}L\omega^{-1}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
       plt.ylim(yaxis2); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.subplot(133); plt.semilogy(base_flow['vzz'],params['z'],'b')
       plt.xlabel(r"$v_{zz}L\omega^{-1}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
       plt.ylim(yaxis3); plt.grid()
       plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
       plt.savefig(plotname,format="png"); plt.close(fig);


     plotname = paths['b_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.5))
     plt.subplot(131); plt.plot(base_flow['b'],params['z'],'b')
     plt.xlabel(r"$bL^{-1}N^{-2}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis1); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132); plt.plot(base_flow['b'],params['z'],'b')
     plt.xlabel(r"bL^{-1}N^{-2}",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
     plt.ylim(yaxis2); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133); plt.semilogy(base_flow['b'],params['z'],'b')
     plt.xlabel(r"bL^{-1}N^{-2}",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis3); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

     plotname = paths['bz_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.5))
     plt.subplot(131); plt.plot(base_flow['bz'],params['z'],'b')
     plt.xlabel(r"$b_zN^{-2}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis1); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132); plt.plot(base_flow['bz'],params['z'],'b')
     plt.xlabel(r"$b_zN^{-2}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
     plt.ylim(yaxis2); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133); plt.semilogy(base_flow['bz'],params['z'],'b')
     plt.xlabel(r"$b_zN^{-2}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis3); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

     plotname = paths['bzz_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.5))
     plt.subplot(131); plt.plot(base_flow['bzz'],params['z'],'b')
     plt.xlabel(r"$b_{zz}LN^{-2}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis1); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132); plt.plot(base_flow['bzz'],params['z'],'b')
     plt.xlabel(r"$b_{zz}LN^{-2}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13) 
     plt.ylim(yaxis2); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133); plt.semilogy(base_flow['bzz'],params['z'],'b')
     plt.xlabel(r"$b_{zz}LN^{-2}$",fontsize=13); plt.ylabel(r"$z/H$",fontsize=13)
     plt.ylim(yaxis3); plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);









  """
  base_flow = ( u / params['U'] , uz /  params['U'] * params['L'] ,
                  uzz /  params['U'] * params['L']**2. , v / params['U'] , 
                  vz /  params['U'] * params['L'] , vzz /  params['U'] * params['L']**2. ,
                  b / (params['N']**2. * params['L']) , bz / (params['N']**2.) ,
                  bzz / (params['N']**2.) * params['L'] )
  """



  """
 

  if plot_flag == 1:
   
   if np.floor(count/freq) == count/freq:
   
     plotname = paths['figure_path'] + paths['u_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.25))
     plt.subplot(131)
     plt.plot(u / params['U'] ,z,'b')
     plt.xlabel(r"u/U",fontsize=13)
     plt.ylabel(r"z/L",fontsize=13)
     plt.xlim([-3.,3.]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132)
     plt.plot(u / params['U'] ,z,'b')
     plt.xlabel(r"u/U",fontsize=13)
     plt.ylabel(r"z/L",fontsize=13)
     plt.axis([-3.,3.,-0.001,0.03]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133)
     plt.semilogy(u / params['U'] ,z,'b')
     plt.xlabel(r"u/U",fontsize=13)
     plt.ylabel(r"z/L",fontsize=13)
     plt.axis([-3.,3.,0.,0.03]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

     plotname = paths['figure_path'] + paths['uz_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.25))
     plt.subplot(131)
     plt.plot(uz / params['omg'] ,z,'b')
     plt.xlabel(r"$u_z/\omega$",fontsize=13)
     plt.ylabel(r"$z/L$",fontsize=13)
     plt.xlim([-500.,500.]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132)
     plt.plot(uz / params['omg'] ,z,'b')
     plt.xlabel(r"$u_z/\omega$",fontsize=13)
     plt.ylabel(r"$z/L$",fontsize=13)
     plt.axis([-500.,500.,-0.001,0.03]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133)
     plt.semilogy(uz / params['omg'] ,z,'b')
     plt.xlabel(r"$u_z/\omega$",fontsize=13)
     plt.ylabel(r"$z/L$",fontsize=13)
     plt.axis([-500.,500.,0.,0.03]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

     plotname = paths['figure_path'] + paths['bz_path'] +'%i.png' %(count)
     fig = plt.figure(figsize=(16,4.25))
     plt.subplot(131)
     plt.plot(bz / (params['N']**2. * np.sin(params['tht'])),z,'b')
     plt.xlabel(r"$b_z/N^2\sin\theta$",fontsize=13)
     plt.ylabel(r"$z/L$",fontsize=13)
     plt.xlim([-200.,200.]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(132)
     plt.plot(bz / (params['N']**2. * np.sin(params['tht']))  ,z,'b')
     plt.xlabel(r"$b_z/N^2\sin\theta$",fontsize=13)
     plt.ylabel(r"$z/L$",fontsize=13)
     plt.axis([-200.,200.,-0.001,0.03]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.subplot(133)
     plt.semilogy(bz / (params['N']**2. * np.sin(params['tht']))  ,z,'b')
     plt.xlabel(r"$b_z/N^2\sin\theta$",fontsize=13)
     plt.ylabel(r"$z/L$",fontsize=13)
     plt.axis([-200.,200.,0.,0.03]) 
     plt.grid()
     plt.title(r"t/T = %.4f, step = %i" %(time,count),fontsize=13)
     plt.savefig(plotname,format="png"); plt.close(fig);

  else:
   
   return 
  """
  return
