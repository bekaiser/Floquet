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
# time-step functions functions


def rk4_time_step( params , Phin , dt, stop_time, case_flag ):
  # uniform time step 4th-order Runge-Kutta time stepper

  time = 0. # non-dimensional time
  count = 0
  output_period = 10
  output_count = 0
  
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
    check_matrix(Phin,'Phin')

  dtf = stop_time - time
  k1 = rk4( params , time , Phin , count , 0 , case_flag )
  k2 = rk4( params , time + dtf/2. , Phin + k1*dtf/2. , count , 0 , case_flag )
  k3 = rk4( params , time + dtf/2. , Phin + k2*dtf/2. , count , 0 , case_flag )
  k4 = rk4( params , time + dtf , Phin + k3*dtf , count , 0 , case_flag )
  Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dtf/6.; 

  final_time = time + dtf # non-dimensional final time
  count = count + 1

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

  if case_flag == 'inviscid_buoyancy':
    A21 = 2.*np.pi*1j*params['k']*np.sin(time)/(1.-params['c2']) #A21 = -2.*np.pi*np.sin(time)
    A22 = 1j*params['k']*np.cos(time)/(params['c2']-1.) #A22 = np.cos(time)
    A = np.matrix([[0.,1.],[A21,A22]],dtype=complex)

  if case_flag == 'forcing_test':
    #A = np.cos(time)
    omg = params['omg']
    A = np.cos(omg*time)



  
  # to use ATLAS BLAS library, both arguments in np.dot should be C-ordered. Check with:
  #print(Am.flags,Phi.flags)
  # Runge-Kutta coefficients
  #start_time_4 = datetime.now()
  krk = np.dot(A,Phin) 
  #time_elapsed = datetime.now() - start_time_4
  #print('A dot Phi time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
 
  check_matrix(krk,'krk')

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
   z = np.linspace((H/Nz)/2. , H, num=Nz) / H 
 if grid_flag == 'cosine': # half cosine grid
   z = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H
   z = z[0:Nz] #/ 2. 
   dz = z[1:Nz]-z[0:Nz-1]
 return z,dz


def nonrotating_solution( params, time ):  
 # don't use, the u & b are 90 degrees out of phase
 
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
   u = u + np.real(U*np.exp(1j*omg*time))
   b = b + np.real(Bs*1j*np.exp(1j*omg*time)) # check if this is correct!

 u = np.real(u) 
 b = np.real(b)
 uz = np.real(uz) 
 bz = np.real(bz)
 
 return  b, u, bz, uz


def rotating_solution( params, time, order ):

 z = params['z']
 U = params['U']
 N = params['N'] 
 omg = params['omg']
 Nz = params['Nz']
 Pr = params['Pr']
 nu = params['nu']
 kap = params['kap']
 L = params['L']
 f = params['f'] 

 tht = params['tht']
 #thtcrit = params['thtc'] #ma.asin(omg/N) # radians 
 #criticality = params['C']

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
 phi1 = beta/(3.*2.**(1./3.)) - (2.)**(1./3.) * (a4**2. + 3.*a2) / (3.*beta) - 1j*a4/3. 
 phi2 = - (1. - 1j*np.sqrt(3.)) * beta / (6.*2.**(1./3.)) +  \
          (1. + 1j*np.sqrt(3.)) * (a4**2. + 3.*a2) / (3.*2**(2./3.)*beta) - 1j*a4/3. 
 phi3 = - (1. + 1j*np.sqrt(3.)) * beta / (6.*2.**(1./3.)) +  \
         (1. - 1j*np.sqrt(3.)) * (a4**2. + 3.*a2) / (3.*2**(2./3.)*beta) - 1j*a4/3. 

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
                     u3*np.exp(-np.sqrt(phi3)*z[i]) + ap*omg ) * np.exp(1j*omg*time) ) / (N**2.*np.sin(tht))
   if order >= 1:
     uz[i] = np.real( ( - np.sqrt(phi1)*u1*np.exp(-np.sqrt(phi1)*z[i]) - np.sqrt(phi2)*u2*np.exp(-np.sqrt(phi2)*z[i]) - \
                        np.sqrt(phi3)*u3*np.exp(-np.sqrt(phi3)*z[i]) ) * np.exp(1j*omg*time) ) / (N**2.*np.sin(tht))
   if order == 2:
     uzz[i] = np.real( ( np.sqrt(phi1)**2.*u1*np.exp(-np.sqrt(phi1)*z[i]) + np.sqrt(phi2)**2.*u2*np.exp(-np.sqrt(phi2)*z[i]) + \
                     + np.sqrt(phi3)**2.*u3*np.exp(-np.sqrt(phi3)*z[i]) ) * np.exp(1j*omg*time) ) / (N**2.*np.sin(tht))
   if f > 0.:
     v[i] = np.real( ( v1*np.exp(-np.sqrt(phi1)*z[i]) + v2*np.exp(-np.sqrt(phi2)*z[i]) + \
                       v3*np.exp(-np.sqrt(phi3)*z[i]) + ap*(omg**2.-(N*np.sin(tht))**2 ) - \
                       A*N**2.*np.sin(tht) ) * 1j * np.exp(1j*omg*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 
     if order >= 1:
       vz[i] = np.real( ( - np.sqrt(phi1)*v1*np.exp(-np.sqrt(phi1)*z[i]) - np.sqrt(phi2)*v2*np.exp(-np.sqrt(phi2)*z[i]) - \
                          np.sqrt(phi3)*v3*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*omg*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 
     if order == 2:
       vzz[i] = np.real( ( np.sqrt(phi1)**2.*v1*np.exp(-np.sqrt(phi1)*z[i]) + np.sqrt(phi2)**2.*v2*np.exp(-np.sqrt(phi2)*z[i]) + \
                         np.sqrt(phi3)**2.*v3*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*omg*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 
   if f <= 0.:
     v[i] = 0.
     if order >= 1:
       vz[i] = 0.
     if order == 2:
       vzz[i] = 0.
   b[i] = np.real( ( c2*np.exp(-np.sqrt(phi1)*z[i]) + c4*np.exp(-np.sqrt(phi2)*z[i]) + \
                     c6*np.exp(-np.sqrt(phi3)*z[i]) + ap ) * 1j * np.exp(1j*omg*time) ) 
   if order >= 1:
     bz[i] = np.real( ( - np.sqrt(phi1)*c2*np.exp(-np.sqrt(phi1)*z[i]) - np.sqrt(phi2)*c4*np.exp(-np.sqrt(phi2)*z[i]) - \
                        np.sqrt(phi3)*c6*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*omg*time) ) 
   if order == 2:
     bzz[i] = np.real( ( np.sqrt(phi1)**2.*c2*np.exp(-np.sqrt(phi1)*z[i]) + np.sqrt(phi2)**2.*c4*np.exp(-np.sqrt(phi2)*z[i]) + \
                      np.sqrt(phi3)**2.*c6*np.exp(-np.sqrt(phi3)*z[i]) ) * 1j * np.exp(1j*omg*time) ) 
 
 if params['wall'] == 'moving':
   u = u + np.real(U*np.exp(1j*omg*time))
   b = b - np.real(ap*1j*np.exp(1j*omg*time)) # why is this the opposite sign from the non-rotating case? Which one is wrong?
   if f > 0.:
     v = v - np.real( ( ap*(omg**2.-(N*np.sin(tht))**2 ) - A*N**2.*np.sin(tht) ) * 1j * np.exp(1j*omg*time) ) / (f * np.cos(tht) * N**2.*np.sin(tht))
     # subtract the geostropphic component of v; means the wall moves in the along-slope direction

 if order < 1.:
   return np.real(b), np.real(u), np.real(v)
 if order == 1.:
   return np.real(b), np.real(u), np.real(v), np.real(bz), np.real(uz), np.real(vz)
 if order == 2.:
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


def partial_zz( params, lower_BC_flag , upper_BC_flag ):
 # second derivative, permiting non-uniform grids
 Nz = params['Nz']
 z = params['z']
 H = params['H']

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


def partial_z( params, lower_BC_flag , upper_BC_flag ):
 # first-order derivative matrix, 2nd order accurate truncation
 Nz = params['Nz']
 H = params['H']
 z = params['z']
 
 # interior points, variable grid
 diagm1 = np.zeros([Nz-1],dtype=complex)
 diag0 = np.zeros([Nz],dtype=complex)
 diagp1 = np.zeros([Nz-1],dtype=complex)
 for j in range(1,Nz-1):
   denom = ( ( z[j+1] - z[j] ) * ( z[j] - z[j-1] ) * ( ( z[j+1] - z[j] ) + ( z[j] - z[j-1] ) ) ) 
   diagm1[j-1] = - ( z[j+1] - z[j] )**2. / denom
   diagp1[j] =   ( z[j] - z[j-1] )**2. / denom
   diag0[j] =  ( ( z[j+1] - z[j] )**2. - ( z[j] - z[j-1] )**2. ) / denom
 pz = np.diag(diagp1,k=1) + np.diag(diagm1,k=-1) + np.diag(diag0,k=0) 

 # lower BC
 l0, l1, l2, l3 = weights( -z[0] , z[0] , z[1] , z[2] , z[0] )
 if lower_BC_flag == 'dirchlet':
   l1 = l1 - l0 # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
 if lower_BC_flag == 'neumann':
   l1 = l1 + l0 # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
 pz[0,0:3] = [ l1 , l2 , l3 ]
 lBC = l0
  
 # upper (far field) BC
 l0, l1, l2, l3 = weights( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , z[Nz-1] )
 if upper_BC_flag == 'dirchlet':
   l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
 pz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 
 return pz,lBC


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


#==============================================================================
# error related functions

def check_matrix(self,string):
 if np.any(np.isnan(self)) == True:
  print('NaN detected in '+ string)
 if np.any(np.isinf(self)) == True:
  print('Inf detected in '+ string)
 return
