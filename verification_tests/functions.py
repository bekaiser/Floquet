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
# functions


def check_matrix(self,string):
 if np.any(np.isnan(self)) == True:
  print('NaN detected in '+ string)
 if np.any(np.isinf(self)) == True:
  print('Inf detected in '+ string)
 return


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

