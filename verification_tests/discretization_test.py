# verifies first and second derivative Lagrange polynomial computation
# Bryan Kaiser

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/path/to/application/app/folder')
import functions as fn

figure_path = './verification_tests/figures/discretization_test/'



# =============================================================================
# loop over Nz resolution Chebyshev node grid

max_exp = 10 # power of two, must be equal to or greater than 5 (maximum N = 2^max)
Nr = np.power(np.ones([max_exp-3])*2.,np.linspace(4.,max_exp,max_exp-3)) # resolution 
Ng = int(np.shape(Nr)[0]) # number of resolutions to try

# case 1:
"""
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
Linf23f = np.zeros([Ng]) # infinity norm, 2nd derivative, uniform grid 
Linf2c3f = np.zeros([Ng]) # infinity norm, 2nd derivative, cosine grid 
Linfp3 = np.zeros([Ng]) # infinity norm, Poisson solution, uniform grid
Linfpc3 = np.zeros([Ng]) # infinity norm, Poisson solution, cosine grid
Linfp3FB = np.zeros([Ng]) # infinity norm, Poisson solution, uniform grid
Linfpc3FB = np.zeros([Ng]) # infinity norm, Poisson solution, cosine grid
Linfp3f = np.zeros([Ng])
Linfpc3f = np.zeros([Ng])
Linfp3FBf = np.zeros([Ng]) # infinity norm, Poisson solution, uniform grid
Linfpc3FBf = np.zeros([Ng]) # infinity norm, Poisson solution, cosine grid

Linf10a = np.zeros([Ng])
Linf10b = np.zeros([Ng])
"""

# case 1:
La = np.zeros([Ng]); Lb = np.zeros([Ng]); Lc = np.zeros([Ng])
Ld = np.zeros([Ng]); Le = np.zeros([Ng]); Lf = np.zeros([Ng])
Lg = np.zeros([Ng]); Lh = np.zeros([Ng]); Li = np.zeros([Ng]);
Lj = np.zeros([Ng]); Lk = np.zeros([Ng]); Ll = np.zeros([Ng]);
"""
Lg = np.zeros([Ng]); Lh = np.zeros([Ng]); Li = np.zeros([Ng])
Lj = np.zeros([Ng]); Lk = np.zeros([Ng]); Ll = np.zeros([Ng])
Lm = np.zeros([Ng]); Ln = np.zeros([Ng]); Lo = np.zeros([Ng])
Lp = np.zeros([Ng]); Lq = np.zeros([Ng]); Lr = np.zeros([Ng])
Ls = np.zeros([Ng]); Lt = np.zeros([Ng]); Lu = np.zeros([Ng])
Lv = np.zeros([Ng]); Lw = np.zeros([Ng]); Lx = np.zeros([Ng])
Ly = np.zeros([Ng]); Lz = np.zeros([Ng]); 
"""

output_plot_no = 4
 
H = 500. #  non-dimensional domain height
Hd = 1. # dimensional grid scale

for n in range(0, Ng): 
      
    Nz = int(Nr[n]) # resolution
    #print('Number of grid points: ',Nz)
 
    dz = H/Nz
    z = np.linspace(dz, Nz*dz, num=Nz)-dz/2. # uniform grid

    zc = -np.cos(((np.linspace(1., 2.*Nz, num=int(2*Nz)))*2.-1.)/(4.*Nz)*np.pi)*H+H
    zc = zc[0:Nz] # half cosine grid
    dzc = zc[1:Nz] - zc[0:Nz-1]



    alpha = 1.5 # lower alpha for more points near bot boundary
    gam = 1./5.
    #print(np.tanh(gam))
    zh = H*np.tanh( gam*np.linspace(Nz/alpha, 0.125, num=int(Nz))) / np.tanh(-gam*((Nz+1)/alpha)) + np.ones([Nz])*H #/ np.tanh( gam ) ( np.ones([Nz]) + 
    dzh = zh[1:Nz] - zh[0:Nz-1]

 
    wall_flag = 'null'
    params = {'H': H, 'Hd': Hd, 'Nz':Nz, 'wall_flag':wall_flag, 
              'z':z, 'dz':dz, 'grid_scale':Hd} # non-dimensional grid for functions
    paramsc = {'H': H, 'Hd': Hd, 'Nz':Nz, 'wall_flag':wall_flag, 
               'z':zc, 'dz':dzc, 'grid_scale':Hd} # non-dimensional grid for functions

    z = z*Hd
    zc = zc*Hd
    #print(np.amax(z))

    if n == 1:
        plotname = figure_path + 'grids.png'
        fig = plt.figure(figsize=(24,16)); 
        plt.subplot(2,3,1)
        plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz, z, 'ob', label=r"centers")
        plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}", fontsize=13)
        plt.ylabel(r"$z$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        plt.legend(loc=2,fontsize=13); 
        plt.subplot(2,3,2)
        plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz, zc, 'ob', label=r"centers")
        plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}", fontsize=13)
        plt.ylabel(r"$z$", fontsize=13); plt.grid()
        plt.title(r"cosine grid N = %i" %(Nz), fontsize=13)
        plt.legend(loc=2, fontsize=13); 
        plt.subplot(2,3,3)
        plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz, zh, 'ob', label=r"centers")
        plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}", fontsize=13)
        plt.ylabel(r"$z$", fontsize=13); plt.grid()
        plt.title(r"hyperbolics sine grid N = %i, $\alpha=1/5$" %(Nz), fontsize=13)
        #plt.ylim([-0.1,10.])
        plt.legend(loc=2, fontsize=13); 

        plt.subplot(2,3,4)
        plt.semilogy(np.linspace(0.5, Nz-0.5, num=Nz)/Nz, z, 'ob', label=r"centers")
        plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}", fontsize=13)
        plt.ylabel(r"$z$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        plt.legend(loc=2,fontsize=13); 
        #plt.ylim([-0.1,10.])
        plt.subplot(2,3,5)
        plt.semilogy(np.linspace(0.5, Nz-0.5, num=Nz)/Nz, zc, 'ob', label=r"centers")
        plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}", fontsize=13)
        plt.ylabel(r"$z$", fontsize=13); plt.grid()
        plt.title(r"cosine grid N = %i" %(Nz), fontsize=13)
        plt.legend(loc=2, fontsize=13);
        #plt.ylim([-0.1,10.]) 
        plt.subplot(2,3,6)
        plt.semilogy(np.linspace(0.5, Nz-0.5, num=Nz)/Nz, zh, 'ob', label=r"centers")
        plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}", fontsize=13)
        plt.ylabel(r"$z$", fontsize=13); plt.grid()
        plt.title(r"hyperbolics sine grid N = %i, $\alpha=1/5$" %(Nz), fontsize=13)
        #plt.ylim([-0.1,10.])
        plt.legend(loc=2, fontsize=13); 

        plt.savefig(plotname,format="png")
        plt.close(fig);

    U0 = 2. # free stream velocity
    m = np.pi/(2.*Hd)
    q = 2.*np.pi/Hd

    # case 1:
    u = np.zeros([Nz,1]); uz = np.zeros([Nz,1]); uzz = np.zeros([Nz,1]); uzzzz = np.zeros([Nz,1])
    u[:,0] = U0*np.sin(m*z) # signal velocity u
    uz[:,0] = U0*m*np.cos(m*z) # du/dz
    uzz[:,0] = -U0*m**2.*np.sin(m*z) # d^2u/dz^2
    uzzzz[:,0] = U0*m**4.*np.sin(m*z)
    uc = np.zeros([Nz,1]); uzc = np.zeros([Nz,1]); uzzc = np.zeros([Nz,1]); uzzzzc = np.zeros([Nz,1])
    uc[:,0] = U0*np.sin(m*zc) 
    uzc[:,0] = U0*m*np.cos(m*zc) 
    uzzc[:,0] = -U0*m**2.*np.sin(m*zc)
    uzzzzc[:,0] = U0*m**4.*np.sin(m*zc)

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

    p = np.zeros([Nz,1]); pz = np.zeros([Nz,1]); pzz = np.zeros([Nz,1]); pzzzz = np.zeros([Nz,1])
    p[:,0] = U0*np.cos(q*z) - U0
    pz[:,0] = -U0*q*np.sin(q*z) 
    pzz[:,0] = -U0*q**2.*np.cos(q*z) 
    #zetazz = np.zeros([Nz,1]); zeta[:,0] = U0*q**4.*np.cos(q*z)
    pzzzz[:,0] = U0*q**4.*np.cos(q*z)
    pc = np.zeros([Nz,1]); pzc = np.zeros([Nz,1]); pzzc = np.zeros([Nz,1]); pzzzzc = np.zeros([Nz,1])
    pc[:,0] = U0*np.cos(q*zc) - U0
    pzc[:,0] = -U0*q*np.sin(q*zc) 
    pzzc[:,0] = -U0*q**2.*np.cos(q*zc)
    pzzzzc[:,0] = U0*q**4.*np.cos(q*zc)
    #zetazzc = np.zeros([Nz,1]); zetac[:,0] = U0*q**4.*np.cos(q*zc)
    """
    p = np.zeros([Nz,1]); pz = np.zeros([Nz,1]); pzz = np.zeros([Nz,1])
    p[:,0] = U0*( np.cos(q*z) - np.cos(2.*q*z) )
    pz[:,0] = -U0*q*( np.sin(q*z) - 2.*np.sin(2.*q*z) )
    pzz[:,0] = -U0*q**2.*( np.cos(q*z) - 4.*np.cos(2.*q*z) )
    pc = np.zeros([Nz,1]); pzc = np.zeros([Nz,1]); pzzc = np.zeros([Nz,1])
    pc[:,0] = U0*( np.cos(q*zc) - np.cos(2.*q*zc) )
    pzc[:,0] = -U0*q*( np.sin(q*zc) - 2.*np.sin(2.*q*zc) )
    pzzc[:,0] = -U0*q**2.*( np.cos(q*zc) - 4.*np.cos(2.*q*zc) )
    """    
     
    m2 = 5*2.*np.pi/(4.*Hd)
    # case 4: zero at z=H at lowest order
    zeta = np.zeros([Nz,1]); zetazz = np.zeros([Nz,1])
    zetac = np.zeros([Nz,1]); zetazzc = np.zeros([Nz,1])
    zeta[:,0] = U0*np.cos(m2*z) #U0*(np.cos(z-np.pi) - np.sin(2.*z))
    zetazz[:,0] = -U0*m2**2.*np.cos(m2*z) #U0*(np.cos(z) + 4.*np.sin(2.*z))
    zetac[:,0] = U0*np.cos(m2*zc) #U0*(np.cos(zc-np.pi) - np.sin(2.*zc))
    zetazzc[:,0] = -U0*m2**2.*np.cos(m2*zc) #U0*(np.cos(zc) + 4.*np.sin(2.*zc))

    if n == output_plot_no:
  
        plotname = figure_path + 'analytical_solutions.png'
        fig = plt.figure(figsize=(18,10))
        plt.subplot(2,4,3)
        plt.plot(p/U0,z,'b',label=r"$\psi$")
        plt.plot(pz/(q*U0),z,'k',label=r"$\psi_z$")
        plt.plot(pzz/(q**2.*U0),z,'--r',label=r"$\psi_{zz}$")
        plt.plot(pzzzz/(q**4.*U0),z,'--b',label=r"$\psi_{zzzz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)
        plt.subplot(2,4,7)
        plt.plot(pc/U0,zc,'b',label=r"$\psi$")
        plt.plot(pzc/(q*U0),zc,'k',label=r"$\psi_z$")
        plt.plot(pzzc/(q**2.*U0),zc,'--r',label=r"$\psi_{zz}$")
        plt.plot(pzzzzc/(q**4.*U0),zc,'--b',label=r"$\psi_{zzzz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)
        plt.subplot(2,4,2)
        plt.plot(b/U0,z,'b',label=r"$b$")
        plt.plot(bz/(q*U0),z,'k',label=r"$b_z$")
        plt.plot(bzz/(q**2.*U0),z,'--r',label=r"$b_{zz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 2",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)
        plt.subplot(2,4,6)
        plt.plot(bc/U0,zc,'b',label=r"$b$")
        plt.plot(bzc/(q*U0),zc,'k',label=r"$b_z$")
        plt.plot(bzzc/(q**2.*U0),zc,'--r',label=r"$b_{zz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 2",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)
        plt.subplot(2,4,1)
        plt.plot(u/U0,z,'b',label=r"$u$")
        plt.plot(uz/(m*U0),z,'k',label=r"$u_z$")
        plt.plot(uzz/(m**2.*U0),z,'--r',label=r"$u_{zz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 1",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)
        plt.subplot(2,4,5)
        plt.plot(uc/U0,zc,'b',label=r"$u$")
        plt.plot(uzc/(m*U0),zc,'k',label=r"$u_z$")
        plt.plot(uzzc/(m**2.*U0),zc,'--r',label=r"$u_{zz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 1",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)
        plt.subplot(2,4,4)
        plt.plot(zeta/U0,z,'b',label=r"$\zeta$")
        plt.plot(zetazz/(U0*m2**2.),z,'--r',label=r"$\zeta_{zz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 4",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)
        plt.subplot(2,4,8)
        plt.plot(zetac/U0,zc,'b',label=r"$\zeta$")
        plt.plot(zetazzc/(U0*m2**2.),zc,'--r',label=r"$\zeta_{zz}$")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 4",fontsize=13)
        plt.grid(); plt.legend(loc=3,fontsize=13)

        plt.savefig(plotname,format="png"); plt.close(fig);

    # case 1:
    """
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
    """

    
    """
    # case 3:
    # 1st derivatives, case 3: (the forward derivative needs no mean information)
    pz0 = np.dot( fn.partial_z( params , 'neumann' , 'neumann'   ) , p ) # uniform grid
    pz0c = np.dot( fn.partial_z( paramsc , 'neumann' , 'neumann'   ) , pc ) # cosine grid 
    # 2nd derivatives, case 3:
    pzz0 = np.dot( fn.partial_zz( params , 'robin' , case3_upper_BC  ) , p ) # uniform grid
    pzz0c = np.dot( fn.partial_zz( paramsc , 'robin' , case3_upper_BC ) , pc ) # cosine grid   (use 'open','dirchlet' for vorticity)
    """

    # 1st derivatives:
    BZ = np.dot( fn.diff_matrix( params , 'neumann' , 'neumann' , diff_order=1 , stencil_size=3 ) , b ) # uniform grid  
    BZC = np.dot( fn.diff_matrix( paramsc , 'neumann' , 'neumann' , diff_order=1 , stencil_size=3 ) , bc ) # cosine grid
    Li[n] = np.amax(abs(bz-BZ)/abs(q*U0)) 
    Lj[n] = np.amax(abs(bzc-BZC)/abs(q*U0))

    # 2nd derivatives:
    BZZ = np.dot( fn.diff_matrix( params , 'neumann' , 'neumann' , diff_order=2 , stencil_size=3 ) , b ) # uniform grid
    BZZC = np.dot( fn.diff_matrix( paramsc , 'neumann' , 'neumann' , diff_order=2 , stencil_size=3 ) , bc ) # cosine grid
    Lk[n] = np.amax(abs(bzz-BZZ)/abs(q**2.*U0)) 
    Ll[n] = np.amax(abs(bzzc-BZZC)/abs(q**2.*U0))

    # test for zeta: specify dirchlet at top and nothing on bottom
    UZZ = np.dot( fn.diff_matrix( params , 'open' , 'dirchlet' , diff_order=2 , stencil_size=3 ) , u ) # uniform grid
    UZZC = np.dot( fn.diff_matrix( paramsc , 'open' ,'dirchlet' , diff_order=2 , stencil_size=3 ) , uc ) # cosine grid
    La[n] = np.amax(abs(uzz-UZZ)/abs(m**2.*U0)) 
    Lb[n] = np.amax(abs(uzzc-UZZC)/abs(m**2.*U0)) 

    # test for psi: thom bottom and dirchlet top 
    PZZ = np.dot( fn.diff_matrix( params , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 ) , p ) # uniform grid
    PZZC = np.dot( fn.diff_matrix( paramsc , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 ) , pc ) # cosine grid
    Lc[n] = np.amax(abs(pzz-PZZ)/abs(q**2.*U0)) 
    Ld[n] = np.amax(abs(pzzc-PZZC)/abs(q**2.*U0)) 

    # second test for psi: get error for inversion
    P =  np.dot( np.linalg.inv( fn.diff_matrix( params , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 ) ) , pzz  ) # uniform grid
    PC = np.dot( np.linalg.inv( fn.diff_matrix( paramsc , 'thom' , 'dirchlet' , diff_order=2 , stencil_size=3 ) ) , pzzc ) # cosine grid 
    Le[n] = np.amax(abs(p-P)/abs(U0)) 
    Lf[n] = np.amax(abs(pc-PC)/abs(U0))

    # test for boundary condition scheme:
    UZZZZ =  np.dot( fn.diff_matrix( params , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) , uzz )  # uniform grid
    UZZZZC = np.dot( fn.diff_matrix( paramsc , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) , uzzc )  # cosine grid 
    Lg[n] = np.amax(abs(uzzzz-UZZZZ)/abs(m**4.*U0)) 
    Lh[n] = np.amax(abs(uzzzzc-UZZZZC)/abs(m**4.*U0))
    # take the 2nd derivative with the streamfunction bcs to get zeta, then take another second derivative to zeta diffusion

 
    # Poisson equation solution: (the backward derivative needs mean information, hence the robin BC)
    """
    p0 =  np.dot( np.linalg.inv( fn.partial_zz(  params , 'robin' , case3_upper_BC ) ) , pzz  ) # uniform grid
    p0c = np.dot( np.linalg.inv( fn.partial_zz( paramsc , 'robin' , case3_upper_BC ) ) , pzzc ) # cosine grid 
    p0f =  np.dot( np.linalg.inv( fn.diff_matrix( params , 'thom' , case3_upper_BC , diff_order=2 , stencil_size=3 ) ) , pzz  ) # uniform grid
    p0cf = np.dot( np.linalg.inv( fn.diff_matrix( paramsc , 'thom' , case3_upper_BC , diff_order=2 , stencil_size=3 ) ) , pzzc ) # cosine grid 
    p0FB =  np.dot( np.linalg.inv( fn.partial_zz(  params , 'robin' , case3_upper_BC ) ) , pzz0  ) # uniform grid
    p0cFB = np.dot( np.linalg.inv( fn.partial_zz( paramsc , 'robin' , case3_upper_BC ) ) , pzz0c ) # cosine grid
    p0FBf =  np.dot( np.linalg.inv( fn.diff_matrix(  params , 'thom' , case3_upper_BC , diff_order=2 , stencil_size=3 ) ) , pzz0f  ) # uniform grid
    p0cFBf = np.dot( np.linalg.inv( fn.diff_matrix( paramsc , 'thom' , case3_upper_BC , diff_order=2 , stencil_size=3 ) ) , pzz0cf ) # cosine grid
    """

    if n == output_plot_no:

        plotname = figure_path + 'second_derivative_solutions_case4.png'
        fig = plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.plot(uzzzz/abs(m**4.*U0),z,'k',label=r"analytical")
        plt.plot(UZZZZ/abs(m**4.*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 4, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(1,2,2)
        plt.plot(uzzzzc/abs(m**4.*U0),zc,'k',label=r"analytical")
        plt.plot(UZZZZC/abs(m**4.*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 4, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.savefig(plotname,format="png"); plt.close(fig); 

        """
        plotname = figure_path + 'poisson_solutions_case3_robin.png'
        fig = plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.plot(p/(U0),z,'k',label=r"analytical")
        plt.plot(p0/(U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,2)
        plt.plot(pc/(U0),zc,'k',label=r"analytical")
        plt.plot(p0c/(U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,3)
        plt.semilogx(abs(p-p0)/abs(U0),z,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.subplot(2,2,4)
        plt.semilogx(abs(pc-p0c)/abs(U0),zc,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.savefig(plotname,format="png"); plt.close(fig);

        plotname = figure_path + 'poisson_solutions_case3_thom.png'
        fig = plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.plot(p/(U0),z,'k',label=r"analytical")
        plt.plot(p0FBf/(U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,2)
        plt.plot(pc/(U0),zc,'k',label=r"analytical")
        plt.plot(p0cFBf/(U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,3)
        plt.semilogx(abs(p-p0FBf)/abs(U0),z,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.subplot(2,2,4)
        plt.semilogx(abs(pc-p0cFBf)/abs(U0),zc,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.savefig(plotname,format="png"); plt.close(fig);

        plotname = figure_path + 'poisson_solutions_case1.png'
        fig = plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.plot(u/(U0),z,'k',label=r"analytical")
        plt.plot(u0/(U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,2)
        plt.plot(uc/(U0),zc,'k',label=r"analytical")
        plt.plot(u0c/(U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,3)
        plt.semilogx(abs(u-u0)/abs(U0),z,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.subplot(2,2,4)
        plt.semilogx(abs(uc-u0c)/abs(U0),zc,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.savefig(plotname,format="png"); plt.close(fig);

        plotname = figure_path + 'first_derivative_solutions.png'
        fig = plt.figure(figsize=(18,20))
        plt.subplot(4,3,1)
        plt.plot(uz/(m*U0),z,'k',label=r"analytical")
        plt.plot(uz0/(m*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,2)
        plt.plot(bz/(q*U0),z,'k',label=r"analytical")
        plt.plot(bz0/(q*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 2, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,3)
        plt.plot(pz/(q*U0),z,'k',label=r"analytical")
        plt.plot(pz0/(q*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,4)
        #plt.plot(abs(uz-uz0)/abs(m*U0), z, 'k') 
        plt.semilogx(abs(uz-uz0)/abs(m*U0), z, 'k') 
        plt.ylabel(r"$z$", fontsize=13)
        plt.title(r"uniform grid, case 1, N = %i" %(Nz), fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,5)
        #plt.plot(abs(bz-bz0)/abs(q*U0), z, 'k') 
        plt.semilogx(abs(bz-bz0)/abs(q*U0), z, 'k') 
        plt.ylabel(r"$z$", fontsize=13)
        plt.title(r"uniform grid, case 2, N = %i" %(Nz), fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,6)
        #plt.plot(abs(pz-pz0)/abs(q*U0), z, 'k') 
        plt.semilogx(abs(pz-pz0)/abs(q*U0), z, 'k') 
        plt.ylabel(r"$z$", fontsize=13)
        plt.title(r"uniform grid, case 3, N = %i" %(Nz), fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,7)
        plt.plot(uzc/(m*U0),zc,'k',label=r"analytical")
        plt.plot(uz0c/(m*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,8)
        plt.plot(bzc/(q*U0),zc,'k',label=r"analytical")
        plt.plot(bz0c/(q*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 2, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,9)
        plt.plot(pzc/(q*U0),zc,'k',label=r"analytical")
        plt.plot(pz0c/(q*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,10)
        #plt.plot(abs(uzc-uz0c)/abs(m*U0),z,'k') 
        plt.semilogx(abs(uzc-uz0c)/abs(m*U0),z,'k') 
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,11)
        #plt.plot(abs(bzc-bz0c)/abs(q*U0),z,'k') 
        plt.semilogx(abs(bzc-bz0c)/abs(q*U0),z,'k') 
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 2, N = %i" %(Nz),fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,3,12)
        #plt.plot(abs(pzc-pz0c)/abs(q*U0),z,'k') 
        plt.semilogx(abs(pzc-pz0c)/abs(q*U0),z,'k') 
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.savefig(plotname,format="png"); plt.close(fig);

        plotname = figure_path + 'second_derivative_solutions_case4.png'
        fig = plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.plot(zetazz/(U0),z,'k',label=r"analytical")
        plt.plot(zetazz0/(U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 4, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,2)
        plt.plot(zetazzc/(U0),zc,'k',label=r"analytical")
        plt.plot(zetazz0c/(U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 4, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=2,fontsize=13)
        plt.subplot(2,2,3)
        plt.semilogx(abs(zetazz-zetazz0)/abs(U0),z,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 4, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.subplot(2,2,4)
        plt.semilogx(abs(zetazzc-zetazz0c)/abs(U0),zc,'k')
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 4, N = %i" %(Nz),fontsize=13)
        plt.grid(); 
        plt.savefig(plotname,format="png"); plt.close(fig); 


        plotname = figure_path + 'second_derivative_solutions.png'
        fig = plt.figure(figsize=(18,20))
        plt.subplot(4,4,1)
        plt.plot(uzz/(m**2.*U0),z,'k',label=r"analytical")
        plt.plot(uzz0/(m**2.*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,2)
        plt.plot(bzz/(q**2.*U0),z,'k',label=r"analytical")
        plt.plot(bzz0/(q**2.*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 2, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,3)
        plt.plot(pzz/(q**2.*U0),z,'k',label=r"analytical")
        plt.plot(pzz0/(q**2.*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, Robin BC N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,4)
        plt.plot(pzz/(q**2.*U0),z,'k',label=r"analytical")
        plt.plot(pzz0f/(q**2.*U0),z,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, Thom BC N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,5)
        #plt.plot(abs(uz-uz0)/abs(m*U0), z, 'k') 
        plt.semilogx(abs(uzz-uzz0)/abs(m**2.*U0), z, 'k') 
        plt.ylabel(r"$z$", fontsize=13)
        plt.title(r"uniform grid, case 1, N = %i" %(Nz), fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,6)
        #plt.plot(abs(bz-bz0)/abs(q*U0), z, 'k') 
        plt.semilogx(abs(bzz-bzz0)/abs(q**2.*U0), z, 'k') 
        plt.ylabel(r"$z$", fontsize=13)
        plt.title(r"uniform grid, case 2, N = %i" %(Nz), fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,7)
        #plt.plot(abs(pz-pz0)/abs(q*U0), z, 'k') 
        plt.semilogx(abs(pzz-pzz0)/abs(q**2.*U0), z, 'k') 
        plt.ylabel(r"$z$", fontsize=13)
        plt.title(r"uniform grid, case 3, Robin BC", fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,8)
        plt.semilogx(abs(pzz-pzz0f)/abs(q**2.*U0),z,'k') #,label=r"analytical")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"uniform grid, case 3, Thom BC",fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,9)
        plt.plot(uzzc/(m**2.*U0),zc,'k',label=r"analytical")
        plt.plot(uzz0c/(m**2.*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,10)
        plt.plot(bzzc/(q**2.*U0),zc,'k',label=r"analytical")
        plt.plot(bzz0c/(q**2.*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 2, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,11)
        plt.plot(pzzc/(q**2.*U0),zc,'k',label=r"analytical")
        plt.plot(pzz0c/(q**2.*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,12)
        plt.plot(pzzc/(q**2.*U0),zc,'k',label=r"analytical")
        plt.plot(pzz0cf/(q**2.*U0),zc,'--r',label=r"computed")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, N = %i" %(Nz),fontsize=13)
        plt.grid(); plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,13)
        #plt.plot(abs(uzc-uz0c)/abs(m*U0),z,'k') 
        plt.semilogx(abs(uzzc-uzz0c)/abs(m**2.*U0),zc,'k') 
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 1, N = %i" %(Nz),fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,14)
        #plt.plot(abs(bzc-bz0c)/abs(q*U0),z,'k') 
        plt.semilogx(abs(bzzc-bzz0c)/abs(q**2.*U0),zc,'k') 
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 2, N = %i" %(Nz),fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,15)
        #plt.plot(abs(pzc-pz0c)/abs(q*U0),z,'k') 
        plt.semilogx(abs(pzzc-pzz0c)/abs(q**2.*U0),zc,'k') 
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, Robin BC",fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.subplot(4,4,16)
        plt.semilogx(abs(pzzc-pzz0cf)/abs(q**2.*U0),zc,'k') #label=r"analytical")
        plt.ylabel(r"$z$",fontsize=13)
        plt.title(r"cosine grid, case 3, Thom BC",fontsize=13)
        plt.grid(); #plt.legend(loc=1,fontsize=13)
        plt.savefig(plotname,format="png"); plt.close(fig);
        """




    """
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
    """

    # case 3:
    """
    Linf13[n] = np.amax(abs(pz-pz0)/abs(q*U0)) 
    Linf1c3[n] = np.amax(abs(pzc-pz0c)/abs(q*U0))
    Linf23[n] = np.amax(abs(pzz-pzz0)/abs(q**2.*U0)) 
    Linf2c3[n] = np.amax(abs(pzzc-pzz0c)/abs(q**2.*U0))
    Linfp3[n] = np.amax(abs(p-p0)/abs(U0)) 
    Linfpc3[n] = np.amax(abs(pc-p0c)/abs(U0))
    Linfp3f[n] = np.amax(abs(p-p0f)/abs(U0))  # Poisson with Thom at z=0, Dirchlet at z=H
    Linfpc3f[n] = np.amax(abs(pc-p0cf)/abs(U0))
    Linfp3FB[n] = np.amax(abs(p-p0FB)/abs(U0)) 
    Linfpc3FB[n] = np.amax(abs(pc-p0cFB)/abs(U0))
    Linfp3FBf[n] = np.amax(abs(p-p0FBf)/abs(U0)) 
    Linfpc3FBf[n] = np.amax(abs(pc-p0cFBf)/abs(U0))
    Linf23f[n] = np.amax(abs(pzz-pzz0f)/abs(q**2.*U0)) 
    Linf2c3f[n] = np.amax(abs(pzzc-pzz0cf)/abs(q**2.*U0))

    Linf10a[n] = np.amax(abs(zetazz-zetazz0)/abs(m2**2.*U0)) # dzz with Dirchlet at z=H, open at z=0
    Linf10b[n] = np.amax(abs(zetazzc-zetazz0c)/abs(m2**2.*U0))
    """

"""
print(Nr)
print(Linf10a) 
print(Linf10b)
print(Linfp3f)
print(Linfpc3f)
"""

plotname = figure_path + 'zeta_test_error.png'
fig = plt.figure(figsize=(8,8))
#plt.subplot(1,3,1)
plt.loglog(Nr,La,'r',label=r"uniform grid")
plt.loglog(Nr,Lb,'b',label=r"cosine grid")
plt.loglog(Nr,(La[0]*0.3/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2nd derivative of $\zeta$ : Dirchlet top & open bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.subplot(1,3,2)
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 2",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,3)
plt.loglog(Nr,Linf13,'r',label=r"uniform")
plt.loglog(Nr,Linf1c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 3",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'psi_test_error.png'
fig = plt.figure(figsize=(8,8))
#plt.subplot(1,3,1)
plt.loglog(Nr,Lc,'r',label=r"uniform grid")
plt.loglog(Nr,Ld,'b',label=r"cosine grid")
plt.loglog(Nr,(Lc[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2nd derivative of $\psi$ : Dirchlet top & Thom bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.subplot(1,3,2)
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 2",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,3)
plt.loglog(Nr,Linf13,'r',label=r"uniform")
plt.loglog(Nr,Linf1c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 3",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path + 'psi_inversion_test_error.png'
fig = plt.figure(figsize=(8,8))
#plt.subplot(1,3,1)
plt.loglog(Nr,Le,'r',label=r"uniform grid")
plt.loglog(Nr,Lf,'b',label=r"cosine grid")
plt.loglog(Nr,(Le[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"Inversion of $\psi$ : Dirchlet top & Thom bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.subplot(1,3,2)
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 2",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,3)
plt.loglog(Nr,Linf13,'r',label=r"uniform")
plt.loglog(Nr,Linf1c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 3",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'boundary_condition_test_error.png'
fig = plt.figure(figsize=(8,8))
#plt.subplot(1,3,1)
plt.loglog(Nr,Lg,'r',label=r"uniform grid")
plt.loglog(Nr,Lh,'b',label=r"cosine grid")
plt.loglog(Nr,(Lh[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"Error of $\partial_{zz}(\partial_{zz}\psi)=\partial_{zz}\zeta$",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.subplot(1,3,2)
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 2",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,3)
plt.loglog(Nr,Linf13,'r',label=r"uniform")
plt.loglog(Nr,Linf1c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 3",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
"""
plt.savefig(plotname,format="png"); plt.close(fig);

"""

check_flag = 0
if Linf10a[2] < 2e-3:
    if Linf10a[3] < 5e-4:
        if Linf10a[4] < 8e-5:
            if Linf10a[5] < 2e-5:
                 if Linf10a[6] < 5e-6:
                     check_flag = check_flag + 1
#print(check_flag)
if Linf10b[2] < 3.1e-3:
    if Linf10b[3] < 7.5e-4:
        if Linf10b[4] < 1.9e-4:
            if Linf10b[5] < 4.7e-5:
                 if Linf10b[6] < 1.2e-5:
                     check_flag = check_flag + 1
#print(check_flag)
if Linfp3f[2] < 1.7e-3:
    if Linfp3f[3] < 4.1e-4:
        if Linfp3f[4] < 1.1e-4:
            if Linfp3f[5] < 2.6e-5:
                 if Linfp3f[6] < 6.3e-6:
                     check_flag = check_flag + 1
#print(check_flag)
if Linfpc3f[2] < 2.5e-3:
    if Linfpc3f[3] < 6.1e-4:
        if Linfpc3f[4] < 1.6e-4:
            if Linfpc3f[5] < 3.8e-5:
                 if Linfpc3f[6] < 9.4e-6:
                     check_flag = check_flag + 1
#print(check_flag)
if check_flag == 4:
     print('\n :) 2nd derivative and Poisson equation discretization \n working properly for resolutions Nz = [64,1024]\n') 
else:
     print('\n ERROR: 2nd derivative and Poisson equation discretization \n not working properly for resolutions Nz = [64,1024]\n') 

plotname = figure_path + 'poisson_error_curves.png'
fig = plt.figure(figsize=(24,8))
plt.subplot(1,3,1)
plt.loglog(Nr,Linfp, 'r', label=r"uniform")
plt.loglog(Nr,Linfpc, 'b', label=r"cosine")
plt.xlabel(r"$N$ grid points", fontsize=13)
plt.ylabel(r"L$_\infty$ error", fontsize=13)
plt.title(r"Dirchlet LBC, Neumann UBC, case 1", fontsize=13) 
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,2)
plt.loglog(Nr,Linfp3,'r',label=r"uniform")
plt.loglog(Nr,Linfpc3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"Robin LBC, Dirchlet UBC, case 3",fontsize=13) 
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,3)
plt.loglog(Nr,Linfp3f,'r',label=r"uniform")
plt.loglog(Nr,Linfpc3f,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"Thom LBC, Dirchlet UBC, case 3",fontsize=13) 
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path + 'forward_backward_error_curves.png'
fig = plt.figure(figsize=(24,8))
plt.subplot(1,3,1)
plt.loglog(Nr,LinfpFB, 'r', label=r"uniform")
plt.loglog(Nr,LinfpcFB, 'b', label=r"cosine")
plt.xlabel(r"$N$ grid points", fontsize=13)
plt.ylabel(r"L$_\infty$ error", fontsize=13)
plt.title(r"Dirchlet LBC, Neumann UBC, case 1", fontsize=13) 
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,2)
plt.loglog(Nr,Linfp3FB,'r',label=r"uniform")
plt.loglog(Nr,Linfpc3FB,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"Robin LBC, Dirchlet UBC, case 3",fontsize=13) 
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,3)
plt.loglog(Nr,Linfp3FBf,'r',label=r"uniform")
plt.loglog(Nr,Linfpc3FBf,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"Thom LBC, Dirchlet UBC, case 3",fontsize=13) 
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);



plotname = figure_path + 'first_derivative_error_curve.png'
fig = plt.figure(figsize=(24,8))
plt.subplot(1,3,1)
plt.loglog(Nr,Linf1,'r',label=r"uniform")
plt.loglog(Nr,Linf1c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 1",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,2)
plt.loglog(Nr,Linf12,'r',label=r"uniform")
plt.loglog(Nr,Linf1c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 2",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,3,3)
plt.loglog(Nr,Linf13,'r',label=r"uniform")
plt.loglog(Nr,Linf1c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 3",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);



plotname = figure_path + 'second_derivative_error_curve.png'
fig = plt.figure(figsize=(30,8))
plt.subplot(1,5,1)
plt.loglog(Nr,Linf2,'r',label=r"uniform")
plt.loglog(Nr,Linf2c,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 1",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,5,2)
plt.loglog(Nr,Linf22,'r',label=r"uniform")
plt.loglog(Nr,Linf2c2,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 2",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,5,3)
plt.loglog(Nr,Linf23,'r',label=r"uniform")
plt.loglog(Nr,Linf2c3,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 3, Robin z=0 BC",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,5,4)
plt.loglog(Nr,Linf23f,'r',label=r"uniform")
plt.loglog(Nr,Linf2c3f,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 3, Thom z=0 BC",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,5,5)
plt.loglog(Nr,Linf10a,'r',label=r"uniform")
plt.loglog(Nr,Linf10b,'b',label=r"cosine")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"case 4, open z=0 BC",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

"""


plotname = figure_path +'discretization_error.png' 
#fig = plt.figure(figsize=(11, 5.5))
fig = plt.figure(figsize=(11, 11))

plt.subplot(221)
plt.loglog(Nr,Le,'-ok',label=r"uniform grid")
plt.loglog(Nr,Lf,'--k',label=r"cosine grid")
plt.loglog(Nr,(Le[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"$L_\infty$",fontsize=15)
plt.title(r"$\psi=\partial_{zz}^{-1}\zeta$ error",fontsize=13) # : Dirchlet top & Thom bottom BCs
plt.grid(); plt.legend(loc=1,fontsize=13)

plt.subplot(222)
plt.loglog(Nr,Lg,'-ok',label=r"uniform grid")
plt.loglog(Nr,Lh,'--k',label=r"cosine grid")
plt.loglog(Nr,(Lh[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"$L_\infty$",fontsize=15)
plt.title(r"$\partial_{zz}\zeta$ error",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)

plt.subplot(223)
plt.loglog(Nr,Li,'-ok',label=r"uniform grid")
plt.loglog(Nr,Lj,'--k',label=r"cosine grid")
plt.loglog(Nr,(Lj[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"$L_\infty$",fontsize=15)
plt.title(r"$\partial_{z}b$ error",fontsize=13) # : Dirchlet top & Thom bottom BCs
plt.grid(); plt.legend(loc=1,fontsize=13)

plt.subplot(224)
plt.loglog(Nr,Lk,'-ok',label=r"uniform grid")
plt.loglog(Nr,Ll,'--k',label=r"cosine grid")
plt.loglog(Nr,(Ll[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"$L_\infty$",fontsize=15)
plt.title(r"$\partial_{zz}b$ error",fontsize=13) # : Dirchlet top & Thom bottom BCs
plt.grid(); plt.legend(loc=1,fontsize=13)


plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.95, hspace=0.25,
                   wspace=0.2)

plt.savefig(plotname,format="png"); plt.close(fig);
