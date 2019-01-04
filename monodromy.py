# 

import h5py
import numpy as np
import math as ma
import cmath as cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chi2
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

figure_path = "./figures/"



# =============================================================================
# functions

def steady_nonrotating_solution( f, N, omg, tht, nu, kap, t, z ):
 # Phillips (1970) / Wunsch (1970) non-rotating steady solution

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap

 d0=((4.*nu**2.)/((N**2.)*(np.sin(tht))**2.))**(1./4.) 

 b0 = np.zeros([Nz,Nt])
 u0 = np.zeros([Nz,Nt])
 #bz0 = np.zeros([Nz,Nt])

 for i in range(0,Nz):
  for j in range(0,Nt):
   Z = z[i]/d0
   u0[i,j] = 2.0*nu/d0*(np.tan(tht)**(-1.))*np.exp(-Z)*np.sin(Z)
   b0[i,j] = (N**2.)*d0*np.cos(tht)*np.exp(-Z)*np.cos(Z)
   #bz0[i,j] = -(N**2.)*d0*np.cos(tht)*np.exp(-Z)*( np.sin(Z) + np.cos(Z) )/d0

 return b0, u0

def xforcing_nonrotating_solution( U, f, N, omg, tht, nu, kap, t, z ):

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

 return  b, bz, u, uz

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


def make_Lap_inv(dz,Nz,K2):
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

def make_r(dz,Nz,C,tht,k):
 # 2nd order accurate truncation
 cottht = np.cos(tht)/np.sin(tht)
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - 1j*k*C**2.
 for j in range(0,Nz-1):
  diagNzm1[j] = cottht*C**2./(2.*dz)
 r = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 r[0,0:3] = [ -C**2.*( 1j*k + 3.*cottht/(2.*dz) ), 2.*C**2.*cottht/dz, -C**2.*cottht/(2.*dz) ] # lower (wall) BC
 r[Nz-1,Nz-3:Nz] = [ C**2.*cottht/(2.*dz), -2.*C**2.*cottht/dz, C**2.*( -1j*k + 3.*cottht/(2.*dz) ) ] # upper (far field) BC
 return r

def make_partial_z(dz,Nz):
 # 2nd order accurate truncation
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(2.*dz)
 pz = np.diag(diagNzm1,k=1) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 pz[0,0:3] = [ -3./(2.*dz), 2./dz, -1./(2.*dz) ] # lower (wall) BC
 pz[Nz-1,Nz-3:Nz] = [ 1./(2.*dz), -2./dz, 3./(2.*dz) ] # upper (far field) BC
 return pz

def make_stationary_matrices(dz,Nz,C,K2,tht,k):
 La_inv = make_Lap_inv(dz,Nz,K2)
 pz = make_partial_z(dz,Nz)
 r = make_r(dz,Nz,C,tht,k)
 P4 = np.dot(La_inv,r)
 return La_inv, pz, P4


def make_DI(dz,Nz,U,k,Re):
 # U needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k*U[j] - 1./Re*(K2 + 2./(dz**2.)) # ,nt[itime]
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(Re*dz**2.)
 DI = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 DI[0,0:4] = [1j*k*U[j] - 1./Re*(K2 - 2./(dz**2.)), -5./(Re*dz**2.), 4./(Re*dz**2.), -1./(Re*dz**2.) ] # lower (wall) BC
 DI[Nz-1,Nz-4:Nz] = [-1./(Re*dz**2.), 4./(Re*dz**2.), -5./(Re*dz**2.), 1j*k*U[j] - 1./Re*(K2 - 2./(dz**2.)) ] # upper (far field) BC
 return DI

def make_D4(dz,Nz,U,k,Re,Pr):
 # U needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k*U[j] - 1./(Re*Pr)*(K2 - 2./(dz**2.))
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(Re*Pr*dz**2.)
 D4 = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 D4[0,0:4] = [1j*k*U[j] - 1./(Re*Pr)*(K2 - 2./(dz**2.)), -5./((Re*Pr)*dz**2.), 4./((Re*Pr)*dz**2.), -1./((Re*Pr)*dz**2.) ] # lower (wall) BC
 D4[Nz-1,Nz-4:Nz] = [-1./((Re*Pr)*dz**2.), 4./((Re*Pr)*dz**2.), -5./((Re*Pr)*dz**2.), 1j*k*U[j] - 1./(Re*Pr)*(K2 - 2./(dz**2.)) ] # upper (far field) BC
 return D4

def make_q(k,Uz):
 # Uz needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k*Uz[j]
 q = np.diag(diagNz,k=0) # no BCs needed
 return q

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


def populate_A(DI, D4, P3, P4, pz, dz, Bz, Uz, cottht, k, l):
 A = np.zeros([int(4*Nz),int(4*Nz)], dtype=complex)
 A11 = DI
 A12 = np.zeros([Nz,Nz])
 A13 = -np.diag(Uz[:],k=0) + 1j*k*P3
 A14 = np.diag(C**2.*np.ones([Nz]),k=0)  + 1j*k*P4
 A21 = np.zeros([Nz,Nz])
 A22 = DI 
 A23 = 1j*l*P3
 A24 = 1j*l*P4
 A31 = np.zeros([Nz,Nz])
 A32 = np.zeros([Nz,Nz])
 A33 = DI - np.dot(pz,P3)
 A34 = np.diag(C**2.*cottht*np.ones([Nz]),k=0) - np.dot(pz,P4)
 A41 = np.identity(Nz)
 A42 = np.zeros([Nz,Nz])
 A43 = -np.diag(Bz[:] + cottht*np.ones([Nz]),k=0)
 A44 = D4
 A1 = np.hstack((A11,A12,A13,A14))
 A2 = np.hstack((A21,A22,A23,A24))
 A3 = np.hstack((A31,A32,A33,A34))
 A4 = np.hstack((A41,A42,A43,A44))
 A = np.vstack((A1,A2,A3,A4))
 if np.any(np.isnan(A)):
  print('NaN detected in A in populate_A()')
 if np.any(np.isinf(A)):
  print('Inf detected in A in populate_A()')
 return A


def inst_construct_A(La_inv,pz,P4,dz,Nz,U,Uz,Bz,k,l,Re,Pr):

 (DI, D4, P3) = make_transient_matrices(dz,Nz,U[:,0],k,Re,Pr,Uz[:,0],La_inv)

 # construct A:
 A = populate_A(DI, D4, P3, P4, pz, dz, Bz[:,0], Uz[:,0], cottht, k, l)

 # tests:
 #print(sum(sum(A[0:Nz,0:Nz]-DI)))
 #print(sum(sum(A[(2*Nz):(3*Nz),(2*Nz):(3*Nz)]-( DI - np.dot(pz,P3)) )))
 #print(sum(sum(A[0:Nz,int(2*Nz):int(3*Nz)]+np.diag(Uz[:,0],k=0) - 1j*k*P3)))
 
 return A

def rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,time,Phi): 
 # constructs A and take the dot product A*Phi

 # instantaneous mean flow solutions at time t
 (B, Bz, U, Uz) = xforcing_nonrotating_solution( Uw, f, N, omg, tht, nu, kap, time, z )
 #(coeffs, z, dz, U, Uz, B, Bz) = inst_solutions(N,T,Uw,nu,Pr,C,time,Nz,H)
 #(coeffs, z, dz, U, Uz, B, Bz) = plug_solutions(N,T,Uw,nu,Pr,C,time,Nz,H) # <-------------------------|||
 
 
 # construction of "A" matrix at time t
 A = inst_construct_A(La_inv,pz,P4,dz,np.shape(z)[0],U,Uz,Bz,k,l,Re = Uw**2./(omg*nu),Pr)
 if np.any(np.isnan(A)):
  print('NaN detected in A in inst_construct_A()')
 if np.any(np.isinf(A)):
  print('Inf detected in A in inst_construct_A()')
 if np.any(np.isnan(Phin)):
  print('NaN detected in Phin in inst_construct_A()')
 if np.any(np.isinf(Phin)):
  print('Inf detected in Phin in inst_construct_A()')

 # Runge-Kutta coefficients
 krk = np.dot(A,Phi) 
 if np.any(np.isnan(krk)):
  print('NaN detected in Runge-Kutta coefficient in rk4()')
 if np.any(np.isinf(krk)):
  print('Inf detected in Runge-Kutta coefficient in rk4()')
 
 return krk




# =============================================================================

# fluid properties
nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity

# flow characteristics
T = 44700.0 # s, M2 tide period
omg = 2.0*np.pi/T # rads/s
f = 1e-4 # 1/s, inertial frequency
N = 1e-3 # 1/s, buoyancy frequency
U = 0.01 # m/s, oscillation velocity amplitude
L = U/omg # m, excursion length
tht = 4.0*np.pi/180. # rads, slope angle

# resolution
H = 10.
Nt = 100 #44700 # total number of frames in movie
Nz = 2000  # number of points in the vertical

# Chebyshev grid
k = np.linspace(1., Nz, num=Nz)
z = -np.cos((k*2.-1.)/(2.*Nz)*np.pi)*H/2.+H/2.
#print(z)

# time series
t = [0.] #np.linspace( 0. , T , num=Nt , endpoint=False )
#print(np.shape(t))
#T0,Z0 = np.meshgrid(t/T,z)

# =============================================================================
# plots

cmap1 = 'seismic'
Nc = 200

# Phillips (1970) / Wunsch (1970) steady non-rotating solution
( b0, u0) = steady_nonrotating_solution( f, N, omg, tht, nu, kap, t, z )
if np.shape(t)[0] == 1:
 zplot( z, H, u0, '/u0_nonrotating.png', 'u0' )
 zplot( z, H, b0, '/b0_nonrotating.png', 'b0' )
 logzplot( z, H, u0, '/u0_nonrotating_log.png', 'u0' )
 logzplot( z, H, b0, '/b0_nonrotating_log.png', 'b0' )
if (np.shape(t)[0]) >= 2:
 T0,Z0 = np.meshgrid(t/T,z)
 contour_plots( T0, Z0, u0, H, Nc, cmap1, '/u0_nonrotating.png', '/u0_nonrotating_zoom.png' )
 contour_plots( T0, Z0, b0, H, Nc, cmap1, '/b0_nonrotating.png', '/b0_nonrotating_zoom.png' )

# non-rotating oscillating solution, Baidulov (2010):
( b, u ) = xforcing_nonrotating_solution( U, f, N, omg, tht, nu, kap, t, z )
if np.shape(t)[0] == 1:
 zplot( z, H, u, '/u_nonrotating.png', 'u' )
 zplot( z, H, b, '/b_nonrotating.png', 'b' )
 logzplot( z, H, u, '/u_nonrotating_log.png', 'u' )
 logzplot( z, H, b, '/b_nonrotating_log.png', 'b' )
if np.shape(t)[0] >= 2:
 T0,Z0 = np.meshgrid(t/T,z)
 contour_plots( T0, Z0, u, H, Nc, cmap1, '/u_nonrotating.png', '/u_nonrotating_zoom.png' )
 contour_plots( T0, Z0, b, H, Nc, cmap1, '/b_nonrotating.png', '/b_nonrotating_zoom.png' )

# total nonrotating: steady + oscillating:
if np.shape(t)[0] == 1:
 zplot( z, H, u0 + u, '/u_nonrotating_total.png', 'u_total' )
 zplot( z, H, b + b0 + N**2.*np.cos(tht), '/b_nonrotating_total.png', 'b_total' )
 logzplot( z, H, u0 + u, '/u_nonrotating_total_log.png', 'u_total' )
 logzplot( z, H, b + b0 + N**2.*np.cos(tht), '/b_nonrotating_total_log.png', 'b_total' )
if np.shape(t)[0] >= 2:
 T0,Z0 = np.meshgrid(t/T,z)
 contour_plots( T0, Z0, u0 + u, H, Nc, cmap1, '/u_nonrotating_total.png', '/u_nonrotating_total_zoom.png' )
 contour_plots( T0, Z0, b + b0 + N**2.*np.cos(tht), H, Nc, cmap1, '/b_nonrotating_total.png', '/b_nonrotating_total_zoom.png' )

# Wunsch (1970) steady rotating solution:
( b0, u0, v0 ) = steady_rotating_solution( f, N, omg, tht, nu, kap, t, z )
if np.shape(t)[0] == 1:
 zplot( z, H, u0, '/u0_rotating.png', 'u0' )
 zplot( z, H, v0, '/v0_rotating.png', 'v0' )
 zplot( z, H, b0, '/b0_rotating.png', 'b0' )
 logzplot( z, H, u0, '/u0_rotating_log.png', 'u0' )
 logzplot( z, H, v0, '/v0_rotating_log.png', 'v0' )
 logzplot( z, H, b0, '/b0_rotating_log.png', 'b0' )
if np.shape(t)[0] >= 2:
 T0,Z0 = np.meshgrid(t/T,z)
 contour_plots( T0, Z0, u0, H, Nc, cmap1, '/u0_rotating.png', '/u0_rotating_zoom.png' )
 contour_plots( T0, Z0, v0, H, Nc, cmap1, '/v0_rotating.png', '/v0_rotating_zoom.png' )
 contour_plots( T0, Z0, b0, H, Nc, cmap1, '/b0_rotating.png', '/b0_rotating_zoom.png' )

# rotating oscillating solution:
( b, u, v ) = xforcing_rotating_solution( U, f, N, omg, tht, nu, kap, t, z )
if np.shape(t)[0] == 1:
 zplot( z, H, u, '/u_rotating.png', 'u' )
 zplot( z, H, v, '/v_rotating.png', 'v' )
 zplot( z, H, b, '/b_rotating.png', 'b' )
 logzplot( z, H, u, '/u_rotating_log.png', 'u' )
 logzplot( z, H, v, '/v_rotating_log.png', 'v' )
 logzplot( z, H, b, '/b_rotating_log.png', 'b' )
if np.shape(t)[0] >= 2:
 T0,Z0 = np.meshgrid(t/T,z)
 contour_plots( T0, Z0, u, H, Nc, cmap1, '/u_rotating.png', '/u_rotating_zoom.png' )
 contour_plots( T0, Z0, v, H, Nc, cmap1, '/v_rotating.png', '/v_rotating_zoom.png' )
 contour_plots( T0, Z0, b, H, Nc, cmap1, '/b_rotating.png', '/b_rotating_zoom.png' )

# total rotating: steady + oscillating:
if np.shape(t)[0] == 1:
 zplot( z, H, u0 + u, '/u_rotating_total.png', 'u_total' )
 zplot( z, H, v0 + v, '/v_rotating_total.png', 'v_total' )
 zplot( z, H, b + b0 + N**2.*np.cos(tht), '/b_rotating_total.png', 'b_total' )
 logzplot( z, H, u0 + u, '/u_rotating_total_log.png', 'u_total' )
 logzplot( z, H, v0 + v, '/v_rotating_total_log.png', 'v_total' )
 logzplot( z, H, b + b0 + N**2.*np.cos(tht), '/b_rotating_total_log.png', 'b_total' )
if np.shape(t)[0] >= 2:
 T0,Z0 = np.meshgrid(t/T,z)
 contour_plots( T0, Z0, u0 + u, H, Nc, cmap1, '/u_rotating_total.png', '/u_rotating_total_zoom.png' )
 contour_plots( T0, Z0, v0 + v, H, Nc, cmap1, '/v_rotating_total.png', '/v_rotating_total_zoom.png' )
 contour_plots( T0, Z0, b + b0 + N**2.*np.cos(tht), H, Nc, cmap1, '/b_rotating_total.png', '/b_rotating_total_zoom.png' )



"""


Nt = int(447000)
tp = np.linspace(0.0, T, num=Nt) # s
dt = tp[1] - tp[0] # s

L = Uw/(2.*np.pi/T) # L = L_excursion
k = 10.*2.*np.pi/L # perturbation wavenumber
l = 10.*2.*np.pi/L
K2 = k**2.+l**2.


(coeffs, z, dz, U, Uz, B, Bz) = inst_solutions(N,T,Uw,nu,Pr,C,0.,Nz,H)
#(coeffs, z, dz, U, Uz, B, Bz) = plug_solutions(N,T,Uw,nu,Pr,C,0.,Nz,H) # <-------------------------|||
omg = coeffs[1]
tht = coeffs[2]
thtcrit = coeffs[3]
Bw = coeffs[5]
cottht = np.cos(tht)/np.sin(tht)
Re = Uw**2./(nu*omg)



 
# initialization:

# initialized submatrices of "A"
DI = np.zeros([Nz,Nz], dtype=complex)
D4 = np.zeros([Nz,Nz], dtype=complex)
La = np.zeros([Nz,Nz], dtype=complex)
q = np.zeros([Nz,Nz], dtype=complex)
r = np.zeros([Nz,Nz], dtype=complex)
pz = np.zeros([Nz,Nz], dtype=complex)

# stationary submatrices of "A" (do this once!)
(La_inv, pz, P4) = make_stationary_matrices(dz,Nz,C,K2,tht,k) 

if np.any(np.isnan(La_inv)):
 print('NaN detected in La_inv')
if np.any(np.isinf(La_inv)):
 print('Inf detected in La_inv')
if np.any(np.isnan(pz)):
 print('NaN detected in pz')
if np.any(np.isinf(pz)):
 print('Inf detected in pz')
if np.any(np.isnan(P4)):
 print('NaN detected in P4')
if np.any(np.isinf(P4)):
 print('Inf detected in P4')


# fundamental solution matrix
Phin = np.identity(int(4*Nz), dtype=complex) 

# time advancement:
for n in range(0,Nt):

 time = tp[n]
 print(time/T)

 # Runge-Kutta, 4th order:
 Phi1 = Phin; t1 = time; 
 k1 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t1,Phi1)
 del Phi1
 
 Phi2 = Phin + k1*(dt/2.); t2 = time + dt/2.;
 k2 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t2,Phi2)
 del Phi2

 Phi3 = Phin + k2*(dt/2.); t3 = time + dt/2.; 
 k3 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t3,Phi3)
 del Phi3

 print(k3)


 Phi4 = Phin + k3*dt; t4 = time + dt; 
 k4 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t4,Phi4)
 del Phi4

 Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dt/6.;

 if np.any(np.isnan(Phin)):
  print('NaN detected in Phi')
  break
 if np.any(np.isinf(Phin)):
  print('NaN detected in Phi')
  break

# eigenvalues, eigenvectors
#(eigval, eigvec) = np.linalg.eig(Phin) 
eigvalues = np.linalg.eigvals(Phin) 

# Floquet exponents
flex = np.log(eigvalues)  
if np.any(np.isnan(flex)):
 print('NaN detected in flex')
if np.any(np.isinf(flex)):
 print('NaN detected in flex')
#print(flex)

# save:
savename = data_path + 'floquet_Re%i_C%i_Pr%i.h5' %(int(Re),int(C),int(Pr)) #'/circulation_%i_%i.h5' %(N0,N1) 
f2 = h5py.File(savename, "w")
dset = f2.create_dataset('T', data=T, dtype='f8')
dset = f2.create_dataset('t', data=tp, dtype='f8') 
dset = f2.create_dataset('dt', data=dt, dtype='f8') 
dset = f2.create_dataset('z', data=z, dtype='f8')
dset = f2.create_dataset('dz', data=dz, dtype='f8')
dset = f2.create_dataset('H', data=H, dtype='f8')
dset = f2.create_dataset('Re', data=Re, dtype='f8')
dset = f2.create_dataset('Pr', data=Pr, dtype='f8')
dset = f2.create_dataset('C', data=C, dtype='f8')
dset = f2.create_dataset('nu', data=nu, dtype='f8')
dset = f2.create_dataset('omg', data=omg, dtype='f8')
dset = f2.create_dataset('U', data=Uw, dtype='f8')
dset = f2.create_dataset('N', data=N, dtype='f8')
dset = f2.create_dataset('tht', data=tht, dtype='f8')
#dset = f2.create_dataset('eigval', data=eigval, dtype='f8')
#dset = f2.create_dataset('eigvec', data=eigvec, dtype='f8')
dset = f2.create_dataset('eigvalues', data=eigvalues, dtype='f8')
dset = f2.create_dataset('flex', data=flex, dtype='f8')
print('\nFloquet multiplier computed and written to file' + savename + '.\n')





# clean up
# add loops over what?
# now a propogator save file

 
# Now do eigenvalue analysis to get the eigenvalues of Phi(T). 
# these eigenvalues are the Floquet multipliers, where the multipliers are defined 
# by the Floquet modes: v(T) = multiplier * v(0). The Floquet exponents 
# are defined as multiplier = exp(sig + i*eta), where sig is the real part of 
# the Floquet exponent.
# if the real part of the Floquet exponent is positive then the Floquet 
# mode grows. The imaginary part of the Floquet exponent influences the frequency of the Floquet mode.


# log(multiplier) = exponent*T

"""


