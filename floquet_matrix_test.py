# verification of the propogator matrix A
# Bryan Kaiser
# 2/4/2018

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



from functions import make_Lap_inv, steady_nonrotating_solution, xforcing_nonrotating_solution, make_d, make_e, make_Lap_inv, make_partial_z, make_DI, make_D4, make_A13, make_A14, make_A34, make_A43, check_matrix, rk4, ordered_prod, time_step, build_A_test

figure_path = "./floquet/figures/"



# =============================================================================

# verification test set up:
z=np.array([0.05,0.1,0.15,0.2,0.25,0.3])
dz = z[1]-z[0]
Nz = np.shape(z)[0]
U = np.array([0.00291545, 0.02332362, 0.0787172 , 0.18658892, 0.36443149,0.62973761])
Uz = np.array([0.17492711, 0.69970845, 1.57434402, 2.79883382, 4.37317784,6.29737609])
Bz = 0.001*Uz
k0 = 1
l0 = 1
C = 1./4.
Re = 1000.
K2 = k0**2.+l0**2.
tht=2.*np.pi/180.
Pr = 2.
cot = np.cos(tht)/np.sin(tht)

# =============================================================================
# analytical discrete solutions (corresponding to the "floquet_primer" document)

# matrix L
L00 = -K2 + 2./(dz**2.)
LNN = -K2 + 2./(dz**2.)
L1 = (1./(dz**2.))*np.ones([Nz-1], dtype=complex)
L0 = -(K2 + 2./(dz**2.))*np.ones([Nz], dtype=complex)
L =  np.diag(L1,k=1) + np.diag(L0,k=0) + np.diag(L1,k=-1) 
L[0,0] = L00
L[0,1] = -5./(dz**2.) # -5.*L1[0]
L[0,2] = 4.*L1[0]
L[0,3] = -L1[0]
L[Nz-1,Nz-4] = -L1[0]
L[Nz-1,Nz-3] = 4.*L1[0]
L[Nz-1,Nz-2] = -5./(dz**2.) #-5.*L1[0]
L[Nz-1,Nz-1] = LNN
L_inv = np.linalg.inv(L)
#print(L,'\n')
#print(L[0,1])

# matrix d
dii = 1j*k0*Uz
d = np.diag(dii,k=0)
#print(d)

# matrix e
e00 = -C**2.*(3.*cot/(2.*dz)+1j*k0)
e1 = (C**2.*cot/(2.*dz))*np.ones([Nz-1], dtype=complex)
eNN = C**2.*(3.*cot/(2.*dz)-1j*k0)
e0 = -(1j*k0*C**2.)*np.ones([Nz], dtype=complex)
e =  np.diag(e1,k=1) + np.diag(e0,k=0) - np.diag(e1,k=-1)
e[0,0] = e00
e[0,1] = 4.*e1[0]
e[0,2] = -e1[0]
e[Nz-1,Nz-1] = eNN
e[Nz-1,Nz-2] = -4.*e1[0]
e[Nz-1,Nz-3] = e1[0]
#print(e)

# DI matrix
D00 = 1j*k0*U[0] + (2./(dz**2.)-K2)/Re
DNN = 1j*k0*U[Nz-1] + (2./(dz**2.)-K2)/Re
D0 = 1j*k0*U - (2./(dz**2.) + K2)/Re * np.ones([Nz], dtype=complex)
D1 = 1./(Re*dz**2.) * np.ones([Nz-1], dtype=complex)
DI =  np.diag(D1,k=1) + np.diag(D0,k=0) + np.diag(D1,k=-1)
DI[0,0] = D00
DI[0,1] = -5.*D1[0]
DI[0,2] = 4.*D1[0]
DI[0,3] = -D1[0]
DI[Nz-1,Nz-1] = DNN
DI[Nz-1,Nz-2] = -5.*D1[0]
DI[Nz-1,Nz-3] = 4.*D1[0]
DI[Nz-1,Nz-4] = -D1[0]
#print(DI)

# D4 matrix
D00 = 1j*k0*U[0] + (2./(dz**2.)-K2)/(Re*Pr)
DNN = 1j*k0*U[Nz-1] + (2./(dz**2.)-K2)/(Re*Pr)
D0 = 1j*k0*U - (2./(dz**2.) + K2)/(Re*Pr) * np.ones([Nz], dtype=complex)
D1 = 1./((Re*Pr)*dz**2.) * np.ones([Nz-1], dtype=complex)
D4 =  np.diag(D1,k=1) + np.diag(D0,k=0) + np.diag(D1,k=-1)
D4[0,0] = D00
D4[0,1] = -5.*D1[0]
D4[0,2] = 4.*D1[0]
D4[0,3] = -D1[0]
D4[Nz-1,Nz-1] = DNN
D4[Nz-1,Nz-2] = -5.*D1[0]
D4[Nz-1,Nz-3] = 4.*D1[0]
D4[Nz-1,Nz-4] = -D1[0]

# partial_z matrix
delta0 = 1./(2.*dz)
delta = delta0 * np.ones([Nz-1], dtype=complex)
partial_z =  np.diag(delta,k=1) - np.diag(delta,k=-1)
partial_z[0,0] = -3*delta0
partial_z[0,1] = 4*delta0
partial_z[0,2] = -delta0
partial_z[Nz-1,Nz-1] = 3*delta0
partial_z[Nz-1,Nz-2] = -4*delta0
partial_z[Nz-1,Nz-3] = delta0

# P3 matrix
P3 = np.dot(L_inv,d)
dzP3 = np.dot(partial_z,P3)
 
# P4 matrix
P4 = np.dot(L_inv,e)
dzP4 = np.dot(partial_z,P4)

# DOT OR ELEMENT MULTIPLY IN P3?

# now total A
A11 = DI
A12 = np.zeros([Nz,Nz],dtype=complex)
A13 = - np.diag(Uz,k=0) + 1j*k0*P3
A14 = 1j*k0*P4 + np.eye(Nz,Nz,0,dtype=complex)*C**2. #np.eye(Nz,Nz,0,dtype=complex)*C**2. + 1j*k0*P4
#print(np.shape(A14))
#print('Here =',1j*k0*P4)
A21 = np.zeros([Nz,Nz],dtype=complex)
A22 = DI
A23 = 1j*l0*P3
A24 = 1j*l0*P4
A31 = np.zeros([Nz,Nz],dtype=complex)
A32 = np.zeros([Nz,Nz],dtype=complex)
A33 = DI - dzP3
A34 = np.eye(Nz,Nz,0,dtype=complex)*C**2.*cot - dzP4
A41 = np.eye(Nz,Nz,0,dtype=complex)
A42 = np.zeros([Nz,Nz],dtype=complex)
A43 = - np.diag(Bz,k=0) - np.diag(cot*np.ones([Nz],dtype=complex),k=0) # C ?
A44 = D4

# =============================================================================
# verification of floquet functions for discrete solutions

L_invf,Lf = make_Lap_inv(dz,Nz,K2)
L_inv_err = np.amax(abs( L_invf - L_inv )) 
L_err = np.amax(abs( Lf - L )) 
print('Laplacian operator infinity norm of error = ',L_err)
print('inverse Laplacian operator infinity norm of error = ',L_inv_err)

#print(np.shape(L))
#print(np.shape(Lf))
for i in range(0,Nz):
  for j in range(0,Nz):
    err = Lf[i,j] - L[i,j] 
    #print('i =',i,' j = ',j,' error = ',err)

df = make_d(k0,Uz,Nz)
d_err =  np.amax(abs( df - d ))
print('d matrix infinity norm of error = ',d_err)

ef = make_e(dz,Nz,C,tht,k0)
e_err =  np.amax(abs( ef - e ))
print('e matrix infinity norm of error = ',e_err)

DIf = make_DI(dz,Nz,U,k0,l0,Re)
DI_err =  np.amax(abs( DIf - DI ))
print('DI matrix infinity norm of error = ',DI_err)

D4f = make_D4(dz,Nz,U,k0,l0,Re,Pr)
D4_err =  np.amax(abs( D4f - D4 ))
print('D4 matrix infinity norm of error = ',D4_err)

partial_zf = make_partial_z(dz,Nz)
partial_z_err =  np.amax(abs( partial_zf - partial_z ))
print('partial z matrix infinity norm of error = ',partial_z_err)

P3f = np.dot(L_invf,df)
P3_err =  np.amax(abs( P3f - P3 ))
print('P3 matrix infinity norm of error = ',P3_err)

P4f = np.dot(L_invf,ef)
P4_err =  np.amax(abs( P4f - P4 ))
print('P4 matrix infinity norm of error = ',P4_err)

dzP3f = np.dot(partial_zf,P3f)
dzP3_err =  np.amax(abs( dzP3f - dzP3 ))
print('dzP3 matrix infinity norm of error = ',dzP3_err)

dzP4f = np.dot(partial_zf,P4f)
dzP4_err =  np.amax(abs( dzP4f - dzP4 ))
print('dzP4 matrix infinity norm of error = ',dzP4_err)


[A11f, A12f, A13f, A14f, A21f, A22f, A23f, A24f, A31f, A32f, A33f, A34f, A41f, A42f, A43f, A44f] = build_A_test( DIf , D4f , k0 , l0 , P3f , P4f , Uz , Bz , tht , C , Nz , dz )

A11_err =  np.amax(abs( A11f - A11 ))
print('A11 matrix infinity norm of error = ',A11_err)

A12_err =  np.amax(abs( A12f - A12 ))
print('A12 matrix infinity norm of error = ',A12_err)

A13_err =  np.amax(abs( A13f - A13 ))
print('A13 matrix infinity norm of error = ',A13_err)

A14_err =  np.amax(abs( A14f - A14 ))
print('A14 matrix infinity norm of error = ',A14_err)


A21_err =  np.amax(abs( A21f - A21 ))
print('A21 matrix infinity norm of error = ',A21_err)

A22_err =  np.amax(abs( A22f - A22 ))
print('A22 matrix infinity norm of error = ',A22_err)

A23_err =  np.amax(abs( A23f - A23 ))
print('A23 matrix infinity norm of error = ',A23_err)

A24_err =  np.amax(abs( A24f - A24 ))
print('A24 matrix infinity norm of error = ',A24_err)


A31_err =  np.amax(abs( A31f - A31 ))
print('A31 matrix infinity norm of error = ',A31_err)

A32_err =  np.amax(abs( A32f - A32 ))
print('A32 matrix infinity norm of error = ',A32_err)

A33_err =  np.amax(abs( A33f - A33 ))
print('A33 matrix infinity norm of error = ',A33_err)

A34_err =  np.amax(abs( A34f - A34 ))
print('A34 matrix infinity norm of error = ',A34_err)


A41_err =  np.amax(abs( A41f - A41 ))
print('A41 matrix infinity norm of error = ',A41_err)

A42_err =  np.amax(abs( A42f - A42 ))
print('A42 matrix infinity norm of error = ',A42_err)

A43_err =  np.amax(abs( A43f - A43 ))
print('A43 matrix infinity norm of error = ',A43_err)

A44_err =  np.amax(abs( A44f - A44 ))
print('A44 matrix infinity norm of error = ',A44_err)
