# functions for Floquet analysis

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import expm
#from scipy.stats import chi2
#from scipy import signal
#from scipy.fftpack import fft, fftshift
#import matplotlib.patches as mpatches
#from matplotlib.colors import colorConverter as cc
from datetime import datetime
import numpy.distutils.system_info as sysinfo
sysinfo.get_info('atlas')


# remove dz from RK4


# =============================================================================    
# time-steppers


def rk4_time_step( params , z , Phin , dt, stop_time ):
  # uniform time step 4th-order Runge-Kutta time stepper

  stat_mat = make_stationary_matrices( z, params['H'] , params['C'] , params['tht'] , params['k0'] , params['l0'] )
  A = np.zeros([int(4*params['Nz']),int(4*params['Nz'])],dtype=complex)

  time = 0.
  count = 0

  while time < stop_time: # add round here

   #start_time_kcoeffs = datetime.now()
   k1 = rk4( params , stat_mat , z, A , time , Phin )
   k2 = rk4( params , stat_mat , z, A , time + dt/2. , Phin + k1*dt/2. )
   k3 = rk4( params , stat_mat , z, A , time + dt/2. , Phin + k2*dt/2. )
   k4 = rk4( params , stat_mat , z, A , time + dt , Phin + k3*dt )
   #time_elapsed = datetime.now() - start_time_kcoeffs
   #print('k coeff time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

   #start_time_Phi_update = datetime.now()
   Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dt/6.; 
   #time_elapsed = datetime.now() - start_time_Phi_update
   #print('Phi update time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

   time = time + dt # non-dimensional time
   count = count + 1
   print('time step = ',count)
   print('time =', time)
   print('||Phi||_inf = ',np.amax(abs(Phin)))

   if np.any(np.isnan(Phin)) == True:
    print('NaN detected')
    return
   if np.any(np.isinf(Phin)) == True:
    print('Inf detected')
    return

  print('RK4 method, final time = ', time)
  return Phin


def make_stationary_matrices( z , H , C , tht , k0 , l0 ): 
 L_inv = make_Lap_inv( z , H , k0**2.+l0**2. ) # inverse of the Laplacian
 P4 = np.dot( L_inv , make_e( z , H , C , tht , k0 ) )
 dzP4 = np.dot( partial_z( z , H , 'neumann' , 'neumann' ) , P4 ) # buoyancy BCs
 pz3 = partial_z( z , H , 'dirchlet' , 'neumann' ) # BCs on wall-normal velocity, 1st derivative
 stat_mat = { 'L_inv':L_inv, 'pz3':pz3, 'P4':P4, 'dzP4':dzP4 } 
 return stat_mat


def rk4( params , stat_mat , z, A , time , Phin ):
 # 4th-order Runge-Kutta functions 

 # non-dimensionalize the base periodic flow:
 ( b, u, bz, uz ) = xforcing_nonrotating_solution( params['U'], params['N'], params['omg'], params['tht'], params['nu'], params['kap'], time, z ) 
 u = u / params['U']
 uz = uz / params['omg']
 bz = bz / ( params['N']**2. * np.sin(params['tht']) )

 #start_time_3 = datetime.now()
 A = fast_A( params , stat_mat , u/params['U'] , uz/params['omg'] , bz/(params['N']**2. * np.sin(params['tht'])) , z , A*(0.+0.j) )
 #time_elapsed = datetime.now() - start_time_3
 #print('build A time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

 # to use ATLAS BLAS library, both arguments in np.dot should be C-ordered. Check with:
 #print(Am.flags)
 #print(Phi.flags)

 # Runge-Kutta coefficients
 #start_time_4 = datetime.now()
 krk = np.dot(A,Phin) 
 #time_elapsed = datetime.now() - start_time_4
 #print('A dot Phi time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
 
 check_matrix(krk,'krk')

 return krk


def fast_A( params , stat_mat , u , uz , bz , z , A ):
 # build the propogator matrix "A"
 
 P3 = np.dot( stat_mat['L_inv'] , make_d( params['k0'] , uz , params['Nz'] ) )
 DI = make_diff_velocity( z , params['H'] , params['k0'] , params['l0'] , params['Re'] , u ) # the advective u must be the time varying component (!)
 D4 = make_diff_buoyancy( z , params['H'] , params['k0'] , params['l0'] , params['Re']*params['Pr'] , u ) 
 Nz = int(params['Nz'])

 # row 1:
 A[0:Nz,0:Nz] = DI
 # A12 = zeros
 A[0:Nz,int(2*Nz):int(3*Nz)] = make_A13( Nz , uz , params['k0'] , P3 )
 A[0:Nz,int(3*Nz):int(4*Nz)] = make_A14( Nz , params['k0'] , stat_mat['P4'] , params['C'] )
 
 # row 2:
 # A21 = zeros
 A[Nz:int(2*Nz),Nz:int(2*Nz)] = DI
 A[Nz:int(2*Nz),int(2*Nz):int(3*Nz)] = 1j * params['l0'] * P3 
 A[Nz:int(2*Nz),int(3*Nz):int(4*Nz)] = 1j * params['l0'] * stat_mat['P4']
 
 # row 3:
 # A31 = zeros
 # A32 = zeros
 A[int(2*Nz):int(3*Nz),int(2*Nz):int(3*Nz)] = DI - np.dot( stat_mat['pz3'] , P3 ) 
 A[int(2*Nz):int(3*Nz),int(3*Nz):int(4*Nz)] = make_A34( Nz , params['C'] , params['tht'] , stat_mat['dzP4'] )

 # row 4:
 A[int(3*Nz):int(4*Nz),0:Nz] = np.eye(Nz,Nz,0,dtype=complex)
 # A42 = zeros
 A[int(3*Nz):int(4*Nz),int(2*Nz):int(3*Nz)] = make_A43( Nz , bz , params['tht'] )
 A[int(3*Nz):int(4*Nz),int(3*Nz):int(4*Nz)] = D4

 return A


def make_diff_velocity( z , H , k0 , l0 , Re , u ):
 # diff = U*i*k - 1/Re*K^2 + 1/Re*dzz 
 Nz = np.shape(z)[0]
 K2 = k0**2.+l0**2.
 diff = partial_zz( z , H , 'dirchlet' , 'neumann' )/Re + np.eye(Nz,Nz,0,dtype=complex)*1j*k0*u - np.eye(Nz,Nz,0,dtype=complex)*K2/Re
 return diff
 

def make_diff_buoyancy( z , H , k0 , l0 , Re , U ):
 Nz = np.shape(z)[0]
 K2 = k0**2.+l0**2.
 diff = partial_zz( z , H , 'neumann' , 'neumann' )/Re + np.eye(Nz,Nz,0,dtype=complex)*1j*k0*U - np.eye(Nz,Nz,0,dtype=complex)*K2/Re
 return diff


def make_Lap_inv( z , H , K2 ): 
 # inverse of the Laplacian, 2nd order accurate truncation, BCs for pressure
 Nz = np.shape(z)[0]
 La = partial_zz( z , H , 'neumann' , 'neumann' ) - np.eye(Nz,Nz,0,dtype=complex)*K2
 La_inv = np.linalg.inv(La)
 return La_inv


def make_e( z , H , C , tht , k0 ):
 # note: this matrix is time-independent (see .pdf document)
 # e = C^2*cot(tht)*dz - C^2*i*k
 cottht = np.cos(tht)/np.sin(tht)
 Nz = np.shape(z)[0]
 # 2nd order accurate truncation, BCs on buoyancy
 e = (C**2.) * cottht * partial_z( z , H , 'neumann' , 'neumann' ) - (C**2. * k0 * 1j) * np.eye(Nz,Nz,0,dtype=complex)
 return e


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


def partial_zz( z , H , lower_BC_flag , upper_BC_flag ):
 # second derivative, permiting non-uniform grids
 Nz = np.shape(z)[0]

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

 # upper (far field) BC
 zj = z[Nz-1] # location of derivative for upper BC
 l0, l1, l2, l3 = weights2( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , zj )
 if upper_BC_flag == 'dirchlet':
   l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
 pzz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 return pzz


def partial_z( z , H , lower_BC_flag , upper_BC_flag ):
 # first-order derivative matrix, 2nd order accurate truncation
 Nz = np.shape(z)[0]
 
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
   
 # upper (far field) BC
 l0, l1, l2, l3 = weights( z[Nz-3] , z[Nz-2] , z[Nz-1] , H + (H-z[Nz-1]) , z[Nz-1] )
 if upper_BC_flag == 'dirchlet':
   l2 = l2 - l3 # Dirchlet phi=0 at z=H (sets phi_ghost = -phi_N)
 if upper_BC_flag == 'neumann':
   l2 = l3 + l2 # Neumann for dz(phi)=0 at z=H (sets phi_ghost = phi_N)
 pz[Nz-1,Nz-3:Nz] = [ l0 , l1 , l2 ]
 
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


def check_matrix(self,string):
 if np.any(np.isnan(self)) == True:
  print('NaN detected in '+ string)
 if np.any(np.isinf(self)) == True:
  print('Inf detected in '+ string)
 return


def make_A43(Nz,Bz,tht):
 cottht = np.cos(tht)/np.sin(tht)
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = Bz[j]
 A43 = - np.diag(diagNz,k=0) - np.eye(Nz,Nz,0,dtype=complex)*cottht
 return A43


def make_A13(Nz,Uz,k,P3):
 #diagNz = np.zeros([Nz], dtype=complex)
 A13 = np.diag(-Uz,k=0) + 1j*k*P3 
 return A13


def make_A34(Nz,C,tht,dzP4):
 cottht = np.cos(tht)/np.sin(tht)
 A34 = np.eye(Nz,Nz,0,dtype=complex)*C**2.*cottht - dzP4 
 return A34


def make_A14(Nz,k0,P4,C):
 A14 = 1j*k0*P4 + np.eye(Nz,Nz,0,dtype=complex)*C**2.
 return A14


def make_d(k0,Uz,Nz):
 # Uz needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k0*float(Uz[j]) # problem with python3.5.0 or less here
 d = np.diag(diagNz,k=0) 
 return d



# =============================================================================    
# analytical flow solutions for constructing the shear and stratification


def steady_nonrotating_solution( N, omg, tht, nu, kap, t, z ):
 # Phillips (1970) / Wunsch (1970) non-rotating steady solution

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap

 d0=((4.*nu**2.)/((N**2.)*(np.sin(tht))**2.))**(1./4.) 

 b0 = np.zeros([Nz,Nt])
 u0 = np.zeros([Nz,Nt])
 bz0 = np.zeros([Nz,Nt])
 uz0 = np.zeros([Nz,Nt])

 for i in range(0,Nz):
  for j in range(0,Nt):
   Z = z[i]/d0
   u0[i,j] = 2.0*nu/(Pr*d0)*(np.tan(tht)**(-1.))*np.exp(-Z)*np.sin(Z)
   b0[i,j] = (N**2.)*d0*np.cos(tht)*np.exp(-Z)*np.cos(Z)
   bz0[i,j] = -(N**2.)*d0*np.cos(tht)*np.exp(-Z)*( np.sin(Z) + np.cos(Z) )/d0
   uz0[i,j] = 2.0*nu/(Pr*d0**2.)*(np.tan(tht)**(-1.))*np.exp(-Z)*(np.cos(Z)-np.sin(Z))

 return b0, u0, bz0, uz0


def xforcing_nonrotating_solution( U, N, omg, tht, nu, kap, t, z ):
 
 t = [t]
 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap
 
 thtcrit = ma.asin(omg/N) # radians 
 criticality = tht/thtcrit
 #print(criticality)

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

 b = np.zeros([Nz,Nt])
 u = np.zeros([Nz,Nt])
 uz = np.zeros([Nz,Nt])
 bz = np.zeros([Nz,Nt])
 
 for i in range(0,Nz):
   for j in range(0,Nt):

     if criticality < 1.:
       u[i,j] = U*np.real( (u1*np.exp(-(1.+1j)*z[i]/d1) + \
                u2*np.exp(-(1.+1j)*z[i]/d2) - 1.)*np.exp(1j*omg*t[j]) )
       uz[i,j] = - U*np.real( (u1*(1.+1j)/d1*np.exp(-(1.+1j)*z[i]/d1) + \
               u2*(1.+1j)/d2*np.exp(-(1.+1j)*z[i]/d2) )*np.exp(1j*omg*t[j]) )
       b[i,j] = Bs*np.real( (b1*np.exp(-(1.0+1j)*z[i]/d1) - \
                b2*np.exp(-(1.+1j)*z[i]/d2) - 1.)*1j*np.exp(1j*omg*t[j]) )
       bz[i,j] = Bs*np.real( ( -(1.0+1j)/d1*b1*np.exp(-(1.0+1j)*z[i]/d1) + \
             (1.+1j)/d2*b2*np.exp(-(1.+1j)*z[i]/d2) )*1j*np.exp(1j*omg*t[j]) )

     if criticality > 1.:
       u[i,j] = U*np.real( (u1*(2.*kap*d1+omg*d1*d2**2.+ \
                1j*(2.*kap*d2-omg*d1**2.*d2))*np.exp((1j-1.0)*z[i]/d2)+ \
                u2*(omg*d1**2.*d2-2.*kap*d2+ \
                1j*(2.0*kap*d1+omg*d1*d2**2.))*np.exp(-(1j+1.)*z[i]/d1)- \
                1.)*np.exp(1j*omg*t[j]) )
       uz[i,j] = U*np.real( ( (1j-1.0)/d2*( alpha1 + 1j*alpha2 )*np.exp((1j-1.0)*z[i]/d2) \
               -(1j+1.)/d1*(alpha3 + 1j*alpha4 )*np.exp(-(1j+1.)*z[i]/d1) )*np.exp(1j*omg*t[j]) )
       b[i,j] = Bs*np.real( (b1*(omg*d1**2.*d2sp-2.*kap*d2+ \
                1j*(2.*kap*d1+omg*d1*d2**2.))*np.exp(-(1j+1.0)*z[j]/d1)+ \
                b2*(2.*kap*d1+omg*d1*d2**2.+ \
                1j*(2.*kap*d2-omg*d1**2.*d2))*np.exp((1j-1.0)*z[i]/d2)- \
                -1.)*1j*np.exp(1j*omg*t[j]) )
       bz[i,j] = Bs*np.real( ( -(1j+1.0)/d1*( beta1 + 1j*beta2 )*np.exp(-(1j+1.0)*z[i]/d1)+ \
               (1j-1.0)/d2*( beta1 + 1j*beta2 )*np.exp((1j-1.0)*z[i]/d2) )*1j*np.exp(1j*omg*t[j]) )

 return  b, u, bz, uz


def steady_rotating_solution( f, N, omg, tht, nu, kap, t, z ):
 # Wunsch (1970) rotating, steady solution

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap
 cot = np.cos(tht) / np.sin(tht)

 d0 = ( ( f*np.cos(tht) / (2.*nu) )**2. + ( N*np.sin(tht)/2. )**2./(nu*kap) )**(-1./4.)
 vg = -Pr*d0*f*cot # geostrophic far field velocity

 b0 = np.zeros([Nz,Nt])
 u0 = np.zeros([Nz,Nt])
 v0 = np.zeros([Nz,Nt])

 for i in range(0,Nz):
  for j in range(0,Nt):
   Z = z[i]/d0
   b0[i,j] = N**2.*d0*np.cos(tht)*np.exp(-Z)*np.cos(Z)
   u0[i,j] = 2.*kap*cot/d0*np.exp(-Z)*np.sin(Z)
   v0[i,j] = Pr*f*d0*cot*np.exp(-Z)*np.cos(Z) + vg

 return b0, u0, v0


def xforcing_rotating_solution( U, f, N, omg, tht, nu, kap, t, z ):

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap
 L = U / omg
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
 
 b = np.zeros([Nz,Nt])
 u = np.zeros([Nz,Nt])
 v = np.zeros([Nz,Nt])

 for i in range(0,Nz):
  for j in range(0,Nt):
   u[i,j] = np.real( ( u1*np.exp(-np.sqrt(phi1)*z[i]) + u2*np.exp(-np.sqrt(phi2)*z[i]) + \
                       u3*np.exp(-np.sqrt(phi3)*z[i]) + ap*omg ) * np.exp(1j*omg*t[j]) ) / (N**2.*np.sin(tht))

   v[i,j] = np.real( ( v1*np.exp(-np.sqrt(phi1)*z[i]) + v2*np.exp(-np.sqrt(phi2)*z[i]) + \
                       v3*np.exp(-np.sqrt(phi3)*z[i]) + ap*(omg**2.-(N*np.sin(tht))**2 ) - \
                       A*N**2.*np.sin(tht) ) * 1j * np.exp(1j*omg*t[j]) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 

   b[i,j] = np.real( ( c2*np.exp(-np.sqrt(phi1)*z[i]) + c4*np.exp(-np.sqrt(phi2)*z[i]) + \
                       c6*np.exp(-np.sqrt(phi3)*z[i]) + ap ) * 1j * np.exp(1j*omg*t[j]) ) 

 return b, u, v


# =============================================================================    
# plot functions


def contour_bounds_2d( self ):
 # self is an array.
 # This function finds the absolute maximum and returns it. 
 cb = 0. # contour bound (such that 0 will be the center of the colorbar)
 if abs(np.amin(self)) > abs(np.amax(self)):
    cb = abs(np.amin(self))
 if abs(np.amax(self)) > abs(np.amin(self)):
    cb = abs(np.amax(self))
 return cb


def zplot( z, H, self, name1, name2 ):
 
 fig = plt.figure() #figsize=(8,4))
 plotname = figure_path + name1 #'/u0_nonrotating.png'
 CP=plt.plot(self,z/H,'b'); 
 plt.xlabel(name2); #plt.legend(loc=4); 
 plt.ylabel('z/H')
 plt.savefig(plotname,format="png"); 
 plt.close(fig)
 
 return


def logzplot( z, H, self, name1, name2 ):
 
 fig = plt.figure() #figsize=(8,4))
 plotname = figure_path + name1 #'/u0_nonrotating.png'
 CP=plt.semilogy(self,z/H,'b'); 
 plt.xlabel(name2); #plt.legend(loc=4); 
 plt.ylabel('z/H')
 plt.savefig(plotname,format="png"); 
 plt.close(fig)
 
 return


def contour_plots( T0, Z0, u0, H, Nc, cmap1, name1, name2 ):

 #u0cb = contour_bounds_2d( u0 )
 #ctrs = np.linspace(-u0cb,u0cb,num=Nc)
 ctrs = np.linspace(np.amin(u0),np.amax(u0),num=Nc)

 fig = plt.figure(figsize=(8,4))
 plotname = figure_path + name1 #'/u0.png'
 CP=plt.contourf(T0,Z0,u0,ctrs,cmap=cmap1); 
 plt.xlabel('t/T'); #plt.legend(loc=4); 
 plt.ylabel('z')
 fig.colorbar(CP)
 plt.savefig(plotname,format="png"); 
 plt.close(fig)

 fig = plt.figure(figsize=(8,4))
 plotname = figure_path + name2 #'/u0_zoom.png'
 CP=plt.contourf(T0,Z0,u0,ctrs,cmap=cmap1); 
 plt.xlabel('t/T'); #plt.legend(loc=4); 
 plt.ylabel('z')
 fig.colorbar(CP)
 plt.axis([0.,1.,0.,H/500.])
 plt.savefig(plotname,format="png"); 
 plt.close(fig)

 return


