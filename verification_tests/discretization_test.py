# first and second derivatives using Lagrange polynomials
# Bryan Kaiser

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/path/to/application/app/folder')
import functions as fn

figure_path = './verification_tests/figures/discretization_test/'

# problem with case 3 second derivative, lower BC

# =============================================================================
# functions

"""
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
   l0, l1, l2, l3 = weights2( -z[0] , z[0] , z[1] , z[2] , zj )
   pzz[0,0:3] = [ l1 - l0 , l2 , l3 ] # Dirchlet phi=0 at z=0 (sets phi_ghost = -phi_0)
 if lower_BC_flag == 'neumann':
   l0, l1, l2, l3 = weights2( -z[0] , z[0] , z[1] , z[2] , zj ) 
   pzz[0,0:3] = [ l1 + l0 , l2 , l3 ] # Neumann for dz(phi)=0 at z=0 (sets phi_ghost = phi_0)
 if lower_BC_flag == 'open':
   l0, l1, l2, l3 = weights2( z[0] , z[1] , z[2] , z[3] , zj )
   pzz[0,0:4] = [ l0 , l1 , l2 , l3 ]
 if lower_BC_flag == 'robin':
   l0, l1, l2, l3 = weights2( -z[1] , -z[0] , z[0] , z[1] , zj ) 
   pzz[0,0:2] = [ l1 + l2 , l3 - l0 ] # combined Neumann and Dirchlet at z = 0
   
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

 return pzz


def partial_z( z , H , lower_BC_flag , upper_BC_flag ):
 # first-order derivative matrix 
 # 2nd order accurate truncation
 Nz = np.shape(z)[0]
 
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
"""

# =============================================================================
# loop over Nz resolution Chebyshev node grid

max_exp = 9 #12 # power of two, must be equal to or greater than 5 (maximum N = 2^max)
Nr = np.power(np.ones([max_exp-3])*2.,np.linspace(4.,max_exp,max_exp-3)) # resolution 
Ng = int(np.shape(Nr)[0]) # number of resolutions to try

# case 1:
Linf1 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid
Linf2 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 
Linfp = np.zeros([Ng]) # infinity norm, Poisson solution, uniform grid
Linfpc = np.zeros([Ng]) # infinity norm, Poisson solution, cosine grid
LinfpFB = np.zeros([Ng]) # infinity norm, Poisson solution, uniform grid
LinfpcFB = np.zeros([Ng]) # infinity norm, Poisson solution, cosine grid

# case 2:
Linf12 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c2 = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid
Linf22 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c2 = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 

# case 3:
Linf13 = np.zeros([Ng]) # infinity norm, 1st derivative, uniform grid
Linf1c3 = np.zeros([Ng]) # infinity norm, 1st derivative, cosine grid
Linf23 = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c3 = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 
Linfp3 = np.zeros([Ng]) # infinity norm, Poisson solution, uniform grid
Linfpc3 = np.zeros([Ng]) # infinity norm, Poisson solution, cosine grid
Linfp3FB = np.zeros([Ng]) # infinity norm, Poisson solution, uniform grid
Linfpc3FB = np.zeros([Ng]) # infinity norm, Poisson solution, cosine grid
 
H = 1.0 # domain height
Hd = H

for n in range(0,Ng): 
  
  Nz = int(Nr[n]) # resolution
  print('Number of grid points: ',Nz)
 
  dz = H/Nz
  z = np.linspace(dz, Nz*dz, num=Nz)-dz/2. # uniform grid

  zc = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H
  zc = zc[0:Nz] # half cosine grid
  dzc = zc[1:Nz] - zc[0:Nz-1]

  wall_flag = 'null'
  params = {'H': H, 'Hd': Hd, 'Nz':Nz, 'wall_flag':wall_flag,'z':z, 'dz':dz}
  paramsc = {'H': H, 'Hd': Hd, 'Nz':Nz, 'wall_flag':wall_flag,'z':zc, 'dz':dzc}

  if n == 0:
   plotname = figure_path + 'uniform_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz,z,'ob',label=r"centers")
   plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}",fontsize=13)
   plt.ylabel(r"$z$",fontsize=13)
   plt.grid()
   plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'cosine_grid.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz,zc,'ob',label=r"centers")
   plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}",fontsize=13)
   plt.ylabel(r"$z$",fontsize=13)
   plt.grid()
   plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

  U0 = 2. # free stream velocity
  m = np.pi/(2.*H)
  q = 2.*np.pi/H

  # case 1:
  u = np.zeros([Nz,1]); uz = np.zeros([Nz,1]); uzz = np.zeros([Nz,1])
  u[:,0] = U0*np.sin(m*z) # signal velocity u
  uz[:,0] = U0*m*np.cos(m*z) # du/dz
  uzz[:,0] = -U0*m**2.*np.sin(m*z) # d^2u/dz^2
  uc = np.zeros([Nz,1]); uzc = np.zeros([Nz,1]); uzzc = np.zeros([Nz,1])
  uc[:,0] = U0*np.sin(m*zc) 
  uzc[:,0] = U0*m*np.cos(m*zc) 
  uzzc[:,0] = -U0*m**2.*np.sin(m*zc)

  # case 2:
  b = np.zeros([Nz,1]); bz = np.zeros([Nz,1]); bzz = np.zeros([Nz,1])
  b[:,0] = U0*np.cos(q*z)
  bz[:,0] = -U0*q*np.sin(q*z) 
  bzz[:,0] = -U0*q**2.*np.cos(q*z) 
  bc = np.zeros([Nz,1]); bzc = np.zeros([Nz,1]); bzzc = np.zeros([Nz,1])
  bc[:,0] = U0*np.cos(q*zc) 
  bzc[:,0] = -U0*q*np.sin(q*zc) 
  bzzc[:,0] = -U0*q**2.*np.cos(q*zc)

  # case 3:
  p = np.zeros([Nz,1]); pz = np.zeros([Nz,1]); pzz = np.zeros([Nz,1])
  p[:,0] = U0*np.cos(q*z) - U0
  pz[:,0] = -U0*q*np.sin(q*z) 
  pzz[:,0] = -U0*q**2.*np.cos(q*z) 
  pc = np.zeros([Nz,1]); pzc = np.zeros([Nz,1]); pzzc = np.zeros([Nz,1])
  pc[:,0] = U0*np.cos(q*zc) - U0
  pzc[:,0] = -U0*q*np.sin(q*zc) 
  pzzc[:,0] = -U0*q**2.*np.cos(q*zc)

  if n == 4:

   plotname = figure_path + 'solutions_uniform_grid_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(p/U0,z,'b',label=r"$b$")
   plt.plot(pz/(q*U0),z,'k',label=r"$b_z$")
   plt.plot(pzz/(q**2.*U0),z,'--r',label=r"$b_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=3,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'solutions_cosine_grid_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(pc/U0,zc,'b',label=r"$b$")
   plt.plot(pzc/(q*U0),zc,'k',label=r"$b_z$")
   plt.plot(pzzc/(q**2.*U0),zc,'--r',label=r"$b_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=3,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'solutions_uniform_grid_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(b/U0,z,'b',label=r"$b$")
   plt.plot(bz/(q*U0),z,'k',label=r"$b_z$")
   plt.plot(bzz/(q**2.*U0),z,'--r',label=r"$b_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=3,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'solutions_cosine_grid_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(bc/U0,zc,'b',label=r"$b$")
   plt.plot(bzc/(q*U0),zc,'k',label=r"$b_z$")
   plt.plot(bzzc/(q**2.*U0),zc,'--r',label=r"$b_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=3,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'solutions_uniform_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(u/U0,z,'b',label=r"$u$")
   plt.plot(uz/(m*U0),z,'k',label=r"$u_z$")
   plt.plot(uzz/(m**2.*U0),z,'--r',label=r"$u_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=6,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'solutions_cosine_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uc/U0,zc,'b',label=r"$u$")
   plt.plot(uzc/(m*U0),zc,'k',label=r"$u_z$")
   plt.plot(uzzc/(m**2.*U0),zc,'--r',label=r"$u_{zz}$")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized analytical solutions",fontsize=13)
   plt.grid(); plt.legend(loc=6,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

  # case 1:
  # 1st derivatives:
  uz0 = np.dot( fn.partial_z( params , 'dirchlet' , 'neumann' ) , u ) # uniform grid  
  uz0c = np.dot( fn.partial_z( paramsc , 'dirchlet' , 'neumann' ) , uc ) # cosine grid
  # 2nd derivatives:
  uzz0 = np.dot( fn.partial_zz( params , 'dirchlet' , 'neumann' ) , u ) # uniform grid
  uzz0c = np.dot( fn.partial_zz( paramsc , 'dirchlet' , 'neumann' ) , uc ) # cosine grid
  # Poisson equation solution:
  u0 =  np.dot( np.linalg.inv( fn.partial_zz(  params , 'dirchlet' , 'neumann' ) ) , uzz  ) # uniform grid
  u0c = np.dot( np.linalg.inv( fn.partial_zz( paramsc , 'dirchlet' , 'neumann' ) ) , uzzc ) # cosine grid
  u0FB =  np.dot( np.linalg.inv( fn.partial_zz(  params , 'dirchlet' , 'neumann' ) ) , uzz0  ) # uniform grid
  u0cFB = np.dot( np.linalg.inv( fn.partial_zz( paramsc , 'dirchlet' , 'neumann' ) ) , uzz0c ) # cosine grid

  # case 2:
  # 1st derivatives:
  bz0 = np.dot( fn.partial_z( params , 'neumann' , 'neumann' ) , b ) # uniform grid  
  bz0c = np.dot( fn.partial_z( paramsc , 'neumann' , 'neumann' ) , bc ) # cosine grid
  # 2nd derivatives:
  bzz0 = np.dot( fn.partial_zz( params , 'neumann' , 'neumann' ) , b ) # uniform grid
  bzz0c = np.dot( fn.partial_zz( paramsc , 'neumann' , 'neumann' ) , bc ) # cosine grid
  # Poisson equation solution: both neumann: ERROR! the matrix is singular

  # case 3:
  # 1st derivatives, case 3: (the forward derivative needs no mean information)
  pz0 = np.dot( fn.partial_z( params , 'neumann' , 'neumann'   ) , p ) # uniform grid
  pz0c = np.dot( fn.partial_z( paramsc , 'neumann' , 'neumann'   ) , pc ) # cosine grid 
  # 2nd derivatives, case 3:
  pzz0 = np.dot( fn.partial_zz( params , 'robin' , 'neumann'   ) , p ) # uniform grid
  pzz0c = np.dot( fn.partial_zz( paramsc , 'robin' , 'neumann' ) , pc ) # cosine grid   (use 'open','dirchlet' for vorticity)
  # Poisson equation solution: (the backward derivative needs mean information, hence the robin BC)
  p0 =  np.dot( np.linalg.inv( fn.partial_zz(  params , 'robin' , 'dirchlet' ) ) , pzz  ) # uniform grid
  p0c = np.dot( np.linalg.inv( fn.partial_zz( paramsc , 'robin' , 'dirchlet' ) ) , pzzc ) # cosine grid 
  p0FB =  np.dot( np.linalg.inv( fn.partial_zz(  params , 'robin' , 'dirchlet' ) ) , pzz0  ) # uniform grid
  p0cFB = np.dot( np.linalg.inv( fn.partial_zz( paramsc , 'robin' , 'dirchlet' ) ) , pzz0c ) # cosine grid

  if n == 5:

   plotname = figure_path + 'computed_poisson_solution_uniform_grid_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(p/(U0),z,'k',label=r"analytical")
   plt.plot(p0/(U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized Poisson solution, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_poisson_solution_cosine_grid_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(pc/(U0),zc,'k',label=r"analytical")
   plt.plot(p0c/(U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized Poisson solution, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_poisson_solution_uniform_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(u/(U0),z,'k',label=r"analytical")
   plt.plot(u0/(U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized Poisson solution, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_poisson_solution_cosine_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uc/(U0),z,'k',label=r"analytical")
   plt.plot(u0c/(U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized Poisson solution, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=2,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_uniform_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uz/(m*U0),z,'k',label=r"analytical")
   plt.plot(uz0/(m*U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_cosine_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uzc/(m*U0),zc,'k',label=r"analytical")
   plt.plot(uz0c/(m*U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);
 
   plotname = figure_path + 'computed_2nd_derivative_uniform_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uzz/(m**2.*U0),z,'k',label=r"analytical")
   plt.plot(uzz0/(m**2.*U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(uzzc/(m**2.*U0),zc,'k',label=r"analytical")
   plt.plot(uzz0c/(m**2.*U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_uniform_grid_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(pzz/(q**2.*U0),z,'k',label=r"analytical")
   plt.plot(pzz0/(q**2.*U0),z,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(pzzc/(q**2.*U0),zc,'k',label=r"analytical")
   plt.plot(pzz0c/(q**2.*U0),zc,'--r',label=r"computed")
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.legend(loc=1,fontsize=13)
   plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_uniform_grid_error_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(uz-uz0)/abs(m*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_cosine_grid_error_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(uzc-uz0c)/abs(m*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_uniform_grid_error_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(uzz-uzz0)/abs(m**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N =%i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid_error_case1.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(uzzc-uzz0c)/abs(m**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_uniform_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bz-bz0)/abs(q*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_cosine_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bzc-bz0c)/abs(q*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_uniform_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bzz-bzz0)/abs(q**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid_error_case2.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(bzzc-bzz0c)/abs(q**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_uniform_grid_error_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(pz-pz0)/abs(q*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_1st_derivative_cosine_grid_error_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(pzc-pz0c)/abs(q*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 1$^{st}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_uniform_grid_error_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(pzz-pzz0)/abs(q**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

   plotname = figure_path + 'computed_2nd_derivative_cosine_grid_error_case3.png'
   fig = plt.figure(figsize=(8,8))
   plt.plot(abs(pzzc-pzz0c)/abs(q**2.*U0),z,'k') 
   plt.ylabel(r"$z$",fontsize=13)
   plt.title(r"normalized 2$^{nd}$ derivative, N = %i" %(Nz),fontsize=13)
   plt.grid(); plt.savefig(plotname,format="png"); plt.close(fig);

  # case 1:
  Linf1[n] = np.amax(abs(uz-uz0)/abs(m*U0)) 
  Linf1c[n] = np.amax(abs(uzc-uz0c)/abs(m*U0))
  Linf2[n] = np.amax(abs(uzz-uzz0)/abs(m**2.*U0)) 
  Linf2c[n] = np.amax(abs(uzzc-uzz0c)/abs(m**2.*U0))
  Linfp[n] = np.amax(abs(u-u0)/abs(U0)) 
  Linfpc[n] = np.amax(abs(uc-u0c)/abs(U0))
  LinfpFB[n] = np.amax(abs(u-u0FB)/abs(U0)) 
  LinfpcFB[n] = np.amax(abs(uc-u0cFB)/abs(U0))

  # case 2:
  Linf12[n] = np.amax(abs(bz-bz0)/abs(q*U0)) 
  Linf1c2[n] = np.amax(abs(bzc-bz0c)/abs(q*U0))
  Linf22[n] = np.amax(abs(bzz-bzz0)/abs(q**2.*U0)) 
  Linf2c2[n] = np.amax(abs(bzzc-bzz0c)/abs(q**2.*U0))

  # case 3:
  Linf13[n] = np.amax(abs(pz-pz0)/abs(q*U0)) 
  Linf1c3[n] = np.amax(abs(pzc-pz0c)/abs(q*U0))
  Linf23[n] = np.amax(abs(pzz-pzz0)/abs(q**2.*U0)) 
  Linf2c3[n] = np.amax(abs(pzzc-pzz0c)/abs(q**2.*U0))
  Linfp3[n] = np.amax(abs(p-p0)/abs(U0)) 
  Linfpc3[n] = np.amax(abs(pc-p0c)/abs(U0))
  Linfp3FB[n] = np.amax(abs(p-p0FB)/abs(U0)) 
  Linfpc3FB[n] = np.amax(abs(pc-p0cFB)/abs(U0))



plotname = figure_path + 'poisson_error_case1.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linfp,'r',label=r"uniform")
plt.loglog(Nr,Linfpc,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1d Poisson equation error, mixed BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'FB_error_case1.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,LinfpFB,'r',label=r"uniform")
plt.loglog(Nr,LinfpcFB,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1d Poisson equation error, mixed BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'poisson_error_case3.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linfp3,'r',label=r"uniform")
plt.loglog(Nr,Linfpc3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1d Poisson equation error, mixed BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'FB_error_case3.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linfp3FB,'r',label=r"uniform")
plt.loglog(Nr,Linfpc3FB,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1d forward-backward error, mixed BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + 'first_derivative_error_case1.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf1,'r',label=r"uniform")
plt.loglog(Nr,Linf1c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1$^{st}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'second_derivative_error_case1.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf2,'r',label=r"uniform")
plt.loglog(Nr,Linf2c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2$^{nd}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + 'first_derivative_error_case2.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1$^{st}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'second_derivative_error_case2.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf22,'r',label=r"uniform")
plt.loglog(Nr,Linf2c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2$^{nd}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + 'first_derivative_error_case3.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf13,'r',label=r"uniform")
plt.loglog(Nr,Linf1c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1$^{st}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'second_derivative_error_case3.png'
fig = plt.figure(figsize=(8,8))
plt.loglog(Nr,Linf23,'r',label=r"uniform")
plt.loglog(Nr,Linf2c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2$^{nd}$ derivative error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
