# functions for Floquet analysis

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chi2
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


# =============================================================================    
# functions


def make_Lap_inv(dz,Nz,K2):
 # inverse of the Laplacian 
 # 2nd order accurate truncation
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - K2 - 2./(dz**2.)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(dz**2.)
 La = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 La[0,0:4] = [ -K2 + 2./(dz**2.), -5./(dz**2.), 4./(dz**2.), -1./(dz**2.) ] # lower (wall) BC
 La[Nz-1,Nz-4:Nz] = [ -1./(dz**2.), 4./(dz**2.), -5./(dz**2.), -K2 + 2./(dz**2.) ] # upper (far field) BC
 La_inv = np.linalg.inv(La)
 return La_inv


def rk4_test( alpha, beta, omg, t, P ):
 A0 = [[-alpha,-np.exp(1j*omg*t)],[0.,-beta]]
 # Runge-Kutta coefficients
 krk = np.dot(A0,P) 
 #check_matrix(krk,'krk')
 return krk


def ordered_prod( alpha, beta, omg, t , dt):
 P0 = np.exp([[-alpha,-np.exp(1j*omg*t)],[0.,-beta]]*np.ones([2,2])*dt)
 return P0


def time_step( Nz, N, omg, tht, nu, kap, U, t, z, dz, l0, k0, Phin , dt, Nt):

  [L_inv, partial_z, P4, dzP4] = make_stationary_matrices(dz,Nz,N*np.sin(tht)/omg,k0**2.+l0**2.,tht,k0)

  t = []
  time = -dt
  count = 0

  while time < 1.: #44700.:
   time = time + dt

   # Runge-Kutta, 4th order: 
   k1 = rk4( Nz, N, omg, tht, nu, kap, U, time , z, dz, l0, k0, Phin , L_inv, partial_z, P4, dzP4 )
   k2 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l0, k0, Phin + k1*dt/2. , L_inv, partial_z, P4, dzP4 )
   k3 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l0, k0, Phin + k2*dt/2. , L_inv, partial_z, P4, dzP4 )
   k4 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt , z, dz, l0, k0, Phin + k3*dt , L_inv, partial_z, P4, dzP4 )

   Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dt/6.; 

   print('time step = ',n+1)
   print('dt = ', dt)
   print('time =', time/44700.)

   if np.any(np.isnan(Phin)) == True:
    print('NaN detected')
    return
   if np.any(np.isinf(Phin)) == True:
    print('Inf detected')
    return
  
  return Phin


def adaptive_time_step( Nz, N, omg, tht, nu, kap, U, t, z, dz, l0, k0, Phin , dt, Nt ):

  [L_inv, partial_z, P4, dzP4] = make_stationary_matrices(dz,Nz,N*np.sin(tht)/omg,k0**2.+l0**2.,tht,k0)

  t = []
  time = -dt
  count = 0

  while time < 1.: #44700.:
   time = time + dt

   # Runge-Kutta, 4th order full time step: 
   k1a = rk4( Nz, N, omg, tht, nu, kap, U, time , z, dz, l0, k0, Phin , L_inv, partial_z, P4, dzP4 )
   k2a = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l0, k0, Phin + k1a*dt/2. , L_inv, partial_z, P4, dzP4 )
   k3a = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l0, k0, Phin + k2a*dt/2. , L_inv, partial_z, P4, dzP4 )
   k4a = rk4( Nz, N, omg, tht, nu, kap, U, time + dt , z, dz, l0, k0, Phin + k3a*dt , L_inv, partial_z, P4, dzP4 )
   Phina = Phin + ( k1a + k2a*2. + k3a*2. + k4a )*dt/6.; 

   # Runge-Kutta, 4th order two half time steps: 
   dtb = dt/2.
   time2 = time + dtb
   # step 1:
   k1b = rk4( Nz, N, omg, tht, nu, kap, U, time , z, dz, l0, k0, Phin , L_inv, partial_z, P4, dzP4 )
   k2b = rk4( Nz, N, omg, tht, nu, kap, U, time + dtb/2. , z, dz, l0, k0, Phin + k1b*dtb/2. , L_inv, partial_z, P4, dzP4 )
   k3b = rk4( Nz, N, omg, tht, nu, kap, U, time + dtb/2. , z, dz, l0, k0, Phin + k2b*dtb/2. , L_inv, partial_z, P4, dzP4 )
   k4b = rk4( Nz, N, omg, tht, nu, kap, U, time + dtb , z, dz, l0, k0, Phin + k3b*dtb , L_inv, partial_z, P4, dzP4 )
   Phinb = Phin + ( k1b + k2b*2. + k3b*2. + k4b )*dtb/6.; 
   # step 2: 
   k1b2 = rk4( Nz, N, omg, tht, nu, kap, U, time2 , z, dz, l0, k0, Phinb , L_inv, partial_z, P4, dzP4 )
   k2b2 = rk4( Nz, N, omg, tht, nu, kap, U, time2 + dtb/2. , z, dz, l0, k0, Phinb + k1b*dtb/2. , L_inv, partial_z, P4, dzP4 )
   k3b2 = rk4( Nz, N, omg, tht, nu, kap, U, time2 + dtb/2. , z, dz, l0, k0, Phinb + k2b*dtb/2. , L_inv, partial_z, P4, dzP4 )
   k4b2 = rk4( Nz, N, omg, tht, nu, kap, U, time2 + dtb , z, dz, l0, k0, Phinb + k3b*dtb , L_inv, partial_z, P4, dzP4 )
   Phinb = Phinb + ( k1b + k2b*2. + k3b*2. + k4b )*dtb/6.; 

   trunc_err = np.amax(abs(Phina-Phinb))/15.
   
   if trunc_err <= 1e-10: # small truncation error: grow time step
     dt = dt*2.
   if trunc_err > 1e-7: # large truncation error: shrink time step
     dt = dt/2.

   # Runge-Kutta, 4th order appropriate time step: 
   k1 = rk4( Nz, N, omg, tht, nu, kap, U, time , z, dz, l0, k0, Phin , L_inv, partial_z, P4, dzP4 )
   k2 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l0, k0, Phin + k1*dt/2. , L_inv, partial_z, P4, dzP4 )
   k3 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt/2. , z, dz, l0, k0, Phin + k2*dt/2. , L_inv, partial_z, P4, dzP4 )
   k4 = rk4( Nz, N, omg, tht, nu, kap, U, time + dt , z, dz, l0, k0, Phin + k3*dt , L_inv, partial_z, P4, dzP4 )
   Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dt/6.; 

   count = count + 1
   print('time step = ',count)
   print('dt = ', dt)
   print('time =', time/44700.)

   if np.any(np.isnan(Phin)) == True:
    print('NaN detected')
    return
   if np.any(np.isinf(Phin)) == True:
    print('Inf detected')
    return
  
  return Phin


def make_e(dz,Nz,C,tht,k0):
 # note: this matrix is time-independent (see .pdf document)
 # 2nd order accurate truncation
 cottht = np.cos(tht)/np.sin(tht)
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - 1j*k0*C**2.
 for j in range(0,Nz-1):
  diagNzm1[j] = cottht*C**2./(2.*dz)
 e = np.diag(diagNzm1,1) + np.diag(diagNz,0) - np.diag(diagNzm1,-1) # tridiagonal
 # now add upper and lower BCs:
 e[0,0:3] = [ -C**2.*( 1j*k0 + 3.*cottht/(2.*dz) ), 2.*C**2.*cottht/dz, -C**2.*cottht/(2.*dz) ] # lower (wall) BC
 e[Nz-1,Nz-3:Nz] = [ C**2.*cottht/(2.*dz), -2.*C**2.*cottht/dz, C**2.*( -1j*k0 + 3.*cottht/(2.*dz) ) ] # upper (far field) BC
 return e


def make_partial_z(dz,Nz):
 # first-order derivative matrix 
 # 2nd order accurate truncation
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(2.*dz)
 pz = np.diag(diagNzm1,k=1) - np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 pz[0,0:3] = [ -3./(2.*dz), 2./dz, -1./(2.*dz) ] # lower (wall) BC
 pz[Nz-1,Nz-3:Nz] = [ 1./(2.*dz), -2./dz, 3./(2.*dz) ] # upper (far field) BC
 return pz


def make_stationary_matrices(dz,Nz,C,K2,tht,k0):
 L_inv = make_Lap_inv(dz,Nz,K2)
 partial_z = make_partial_z(dz,Nz)
 #e = make_e(dz,Nz,C,tht,k0)
 P4 = np.dot(L_inv,make_e(dz,Nz,C,tht,k0))
 dzP4 = np.dot(partial_z,P4)
 return L_inv, partial_z, P4, dzP4


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
 diagNz = np.zeros([Nz], dtype=complex)
 A13 = np.diag(-Uz,k=0) + 1j*k*P3 
 return A13


def make_A34(Nz,C,tht,dzP4):
 cottht = np.cos(tht)/np.sin(tht)
 A34 = np.eye(Nz,Nz,0,dtype=complex)*C**2.*cottht - dzP4 
 return A34


def make_A14(Nz,k0,P4,C):
 A14 = 1j*k0*P4 + np.eye(Nz,Nz,0,dtype=complex)*C**2.
 return A14


def make_D4(dz,Nz,U,k0,l0,Re,Pr):
 # 2nd order accurate truncation
 K2 = k0**2.+l0**2.
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - (K2 + 2./(dz**2.))/(Re*Pr)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./((Re*Pr)*dz**2.)
 D =  np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1) 
 # now add upper and lower BCs:
 D[0,0:4] = [ (2./(dz**2.)-K2)/(Re*Pr), -5./((Re*Pr)*dz**2.), 4./((Re*Pr)*dz**2.), -1./((Re*Pr)*dz**2.) ]  # lower (wall) BC
 D[Nz-1,Nz-4:Nz] = [ -1./((Re*Pr)*dz**2.), 4./((Re*Pr)*dz**2.), -5./((Re*Pr)*dz**2.), (2./(dz**2.)-K2)/(Re*Pr) ]  # upper (far field) BC
 D4 = np.eye(Nz,Nz,0,dtype=complex)*1j*k0*U + D
 return D4


def make_DI(dz,Nz,U,k0,l0,Re):
 # 2nd order accurate truncation
 K2 = k0**2.+l0**2.
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - (K2 + 2./(dz**2.))/Re
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(Re*dz**2.)
 D =  np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1) 
 # now add upper and lower BCs:
 D[0,0:4] = [ (2./(dz**2.)-K2)/Re, -5./(Re*dz**2.), 4./(Re*dz**2.), -1./(Re*dz**2.) ]  # lower (wall) BC
 D[Nz-1,Nz-4:Nz] = [ -1./(Re*dz**2.), 4./(Re*dz**2.), -5./(Re*dz**2.), (2./(dz**2.)-K2)/Re ]  # upper (far field) BC
 DI = np.eye(Nz,Nz,0,dtype=complex)*1j*k0*U + D
 return DI


def make_d(k0,Uz,Nz):
 # Uz needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k0*float(Uz[j]) # problem with python3.5.0 or less here
 d = np.diag(diagNz,k=0) 
 return d


def make_transient_matrices(dz,Nz,U,k,Re,Pr,Uz,La_inv):
 DI = make_DI(dz,Nz,U,k,Re)
 D4 = make_D4(dz,Nz,U,k,Re,Pr)
 q = make_q(k,Uz)
 P3 = np.dot(La_inv,q)
 if np.any(np.isnan(DI)):
  print('NaN detected in DI')
 if np.any(np.isinf(DI)):
  print('Inf detected in DI')
 if np.any(np.isnan(D4)):
  print('NaN detected in D4')
 if np.any(np.isinf(D4)):
  print('Inf detected in D4')
 if np.any(np.isnan(q)):
  print('NaN detected in q')
 if np.any(np.isinf(q)):
  print('Inf detected in q')
 if np.any(np.isnan(P3)):
  print('NaN detected in P3')
 if np.any(np.isinf(P3)):
  print('Inf detected in P3')
 return DI, D4, P3


def build_A( DI , D4 , k0 , l0 , P3 , P4 , uz , bz , tht , C , Nz , dz , partial_z , dzP4 ): 

 #partial_z = make_partial_z(dz,Nz)
 dzP3 = np.dot(partial_z,P3)
 #dzP4 = np.dot(partial_z,P4)

 A11 = DI
 A12 = np.zeros([Nz,Nz],dtype=complex) 
 A13 = make_A13(Nz,uz,k0,P3) 
 A14 = make_A14(Nz,k0,P4,C) 
 A21 = np.zeros([Nz,Nz],dtype=complex)
 A22 = DI
 A23 = 1j*l0*P3
 A24 = 1j*l0*P4
 A31 = np.zeros([Nz,Nz],dtype=complex)
 A32 = np.zeros([Nz,Nz],dtype=complex)
 A33 = DI - dzP3
 A34 = make_A34(Nz,C,tht,dzP4)
 A41 = np.eye(Nz,Nz,0,dtype=complex)
 A42 = np.zeros([Nz,Nz],dtype=complex)
 A43 = make_A43(Nz,bz,tht) 
 A44 = D4

 A1 = np.concatenate((A11,A12,A13,A14),axis=1)
 A2 = np.concatenate((A21,A22,A23,A24),axis=1)
 A3 = np.concatenate((A31,A32,A33,A34),axis=1)
 A4 = np.concatenate((A41,A42,A43,A44),axis=1)
 Am = np.concatenate((A1,A2,A3,A4),axis=0)

 return Am 

def rk4( Nz, N, omg, tht, nu, kap, U, t, z, dz, l0, k0, Phi , L_inv, partial_z, P4, dzP4 ):
 
 Lbl = U/omg
 Re = omg*Lbl**2./nu
 Pr = nu/kap
 #print(t[n])

 # Phillips (1970) / Wunsch (1970) steady non-rotating solution
 #( b0, u0, bz0, uz0 ) = steady_nonrotating_solution( N, omg, tht, nu, kap, [t[n]], z )
 # non-rotating oscillating solution, Baidulov (2010):
 ( b, u, bz, uz ) = xforcing_nonrotating_solution( U, N, omg, tht, nu, kap, t, z ) 

 # pressure matrices:
 #d = make_d(k0,uz,Nz)
 #e = make_e(dz,Nz,N*np.sin(tht)/omg,tht,k0)
 #La_inv = make_Lap_inv(dz,Nz,l0**2.+k0**2.)
 P3 = np.dot(L_inv,make_d(k0,uz,Nz))
 #P4 = np.dot(La_inv,e)
 """
 print('e =',e)
 print('d =',d)
 print('La_inv =',La_inv)
 print('partial_z =',partial_z)
 print('P3 =',P3)
 print('P4 =',P4)
 print('dzP4 =',dzP4)
 print('dzP3 =',dzP3)
 """

 #check_matrix(e,'e')
 #check_matrix(d,'d')
 #check_matrix(L_inv,'L_inv')
 check_matrix(P3,'P3')
 check_matrix(P4,'P4')

 # advective-diffusive matrices:
 DI = make_DI(dz,Nz,U,k0,l0,Re)
 D4 = make_D4(dz,Nz,U,k0,l0,Re,Pr)
 """
 print('DI =',DI)
 print('D4 =',D4)
 """

 Am = build_A( DI , D4 , k0 , l0 , P3 , P4 , uz , bz , tht , N*np.sin(tht)/omg , Nz , dz , partial_z , dzP4 )

 check_matrix(Am,'Am')
 check_matrix(Phi,'Phi')

 # Runge-Kutta coefficients
 krk = np.dot(Am,Phi) 
 
 check_matrix(krk,'krk')

 return krk


def ordered_prod( alpha, beta, omg, t , dt):
 P0 = np.exp([[-alpha,-np.exp(1j*omg*t)],[0.,-beta]]*np.ones([2,2])*dt)
 return P0


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


