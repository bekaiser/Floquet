
# Bryan Kaiser

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/path/to/application/app/folder')
import functions as fn

figure_path = './verification_tests/figures/discretization_test/'



# all that I need to work properly is zeta_wall (does!), dzz_zeta, and the inversion!




# =============================================================================
# loop over Nz resolution Chebyshev node grid

max_exp = 12 # power of two, must be equal to or greater than 5 (maximum N = 2^max)
Nr = np.power(np.ones([max_exp-3])*2.,np.linspace(4.,max_exp,max_exp-3)) # resolution 
Ng = int(np.shape(Nr)[0]) # number of resolutions to try

L_zeta_wall = np.zeros([Ng])
L_zeta_wall2 = np.zeros([Ng])

L_psi_zz = np.zeros([Ng])

L_zeta_zz = np.zeros([Ng])

L_psi = np.zeros([Ng])

H = 32. # = Hd/dS, non-dimensional grid height
nu = 1e-6
omg = 2.*np.pi/44700.
dS = np.sqrt(2.*nu/omg)
Hd = H*dS 
wall_BC_off_flag = ' ' 



for n in range(0, Ng): 
      
    Nz = int(Nr[n]) # resolution
    #print('Number of grid points: ',Nz)
 
    grid_flag = 'uniform' #'uniform 2'
    z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid

    if n == 5:
        plotname = figure_path + 'grid.png'
        fig = plt.figure(figsize=(8,8)); 
        plt.subplot(1,1,1)
        plt.plot(np.linspace(0.5, Nz-0.5, num=Nz)/Nz, z/H, 'ob', label=r"centers")
        plt.xlabel(r"$i^{th}$ grid point divided by N where i={1,N}", fontsize=13)
        plt.ylabel(r"$z$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        plt.legend(loc=2,fontsize=13); 
        plt.axis([-0.005,1./H,-0.005,1./H])
        plt.savefig(plotname,format="png")
        plt.close(fig);

    # test signals:
    #u = np.zeros([Nz,1]); uz = np.zeros([Nz,1]); uzz = np.zeros([Nz,1]);
    """
    u = np.sin(2.*np.pi/H*z)-z*(2.*np.pi/H)
    uz = 2.*np.pi/H*np.cos(2.*np.pi/H*z)-np.ones(np.shape(z))*(2.*np.pi/H)
    uzz = -np.sin(2.*np.pi/H*z)*(2.*np.pi/H)**2.
    
    u = np.cos(2.*np.pi/H*z)
    uz = -2.*np.pi/H*np.sin(2.*np.pi/H*z)
    uzz = -np.cos(2.*np.pi/H*z)*(2.*np.pi/H)**2.
    
    u = np.cos(2.*np.pi/H*z)-np.ones(np.shape(z))
    uz = -2.*np.pi/H*np.sin(2.*np.pi/H*z)
    uzz = -np.cos(2.*np.pi/H*z)*(2.*np.pi/H)**2.
    """

    
    """
    u =  ( (z+1.)**3.-z**2. - 3.*z -1. )*np.exp(-10.*H*z)
    uz = -z*(10.*H*z**2.+(20.*H-3.)*z-4.)*np.exp(-10.*H*z)
    uzz = (100.*H**2.*z**3.+20.*H*(10.*H-3.)*z**2.+(6.-80.*H)*z+4.)*np.exp(-10.*H*z)
    """

    m = 0.075 #25
    psi =  ( (z+1.)**3.-z**2. - 3.*z -1. )*np.exp(-m*H*z)
    psi_z = -z*(m*H*z**2.+(2*m*H-3.)*z-4.)*np.exp(-m*H*z)
    psi_zz = ((m*H)**2.*z**3.+2*m*H*(m*H-3.)*z**2.+(6.-8.*m*H)*z+4.)*np.exp(-m*H*z)
    psi_zzzz = ( (H*m)**4.*(-(z**2.)+(z+1.)**3.-3.*z-1.) -(4.*(H*m)**3.)*(3.*(z+1.)**2.-2.*z-3.) + (6.*(H*m)**2.)*(6.*(z+1.)-2.)-(24.*H*m) )*np.exp(-m*H*z)
  
    psi_wall = ( (0.+1.)**3.-0.**2. - 3.*0. -1. )*np.exp(-m*H*0.)
    psi_z_wall =  -0.*(m*H*0.**2.+(2*m*H-3.)*0.-4.)*np.exp(-m*H*0.)
    zeta_wall = ((m*H)**2.*0.**3.+2*m*H*(m*H-3.)*0.**2.+(6.-8.*m*H)*0.+4.)*np.exp(-m*H*0.)
    #print(psi_wall,psi_z_wall,zeta_wall)

    # add high order extrapolator to check:
    #print(psi[Nz-3:Nz-1],psi_z[Nz-3:Nz-1],(psi[Nz-1]-psi[Nz-2])/(2.*z[Nz-1]-z[Nz-2]))

    if n == Ng-1:
        plotname = figure_path + 'solutions.png'
        fig = plt.figure(figsize=(32,8)); 
        plt.subplot(1,4,1)
        plt.plot(psi, z/H, 'b') #, label=r"u")
        #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
        plt.xlabel(r"$\psi$", fontsize=13)
        plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
        #plt.ylim([-0.001,0.05])
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        #plt.legend(loc=2,fontsize=13); 
        #plt.axis([0.,20./H,0.,1./(2.*H)])
        plt.subplot(1,4,2)
        plt.plot(psi_z, z/H, 'b') #, label=r"analytical")
        #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
        plt.xlabel(r"$\partial_{z}\psi$", fontsize=13)
        plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        plt.subplot(1,4,3)
        plt.plot(psi_zz, z/H, 'b') #, label=r"analytical")
        #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
        plt.xlabel(r"$\zeta=\partial_{zz}\psi$", fontsize=13)
        plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        plt.subplot(1,4,4)
        plt.plot(psi_zzzz, z/H, 'b') #, label=r"analytical")
        #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
        plt.xlabel(r"$\partial_{zz}\zeta=\partial_{zzzz}\psi$", fontsize=13)
        plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        #plt.legend(loc=2,fontsize=13); 
        plt.savefig(plotname,format="png")
        plt.close(fig);



    # do zeta_wall error:
    zeta_wallC = psi[0]*(2./(z[0]**2.)) # works for variable grid
    zeta_wallE = abs(zeta_wall-zeta_wallC)/abs(zeta_wall)*100.
    L_zeta_wall[n] = zeta_wallE

    zeta_wallC2 = fn.extrapolate_to_zero( psi_zz , z , 6 )
    zeta_wallE2 = abs(zeta_wall-zeta_wallC2)/abs(zeta_wall)*100.
    L_zeta_wall2[n] = zeta_wallE2


    # now do exactly as in the solutions:
    wall_BC_flag = 'BC'
    grid_params_dzz = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
    grid_params_inv = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
 
    dzz_zeta,lzeta = fn.diff_matrix( grid_params_dzz , 'dirchlet 2' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
    dzz_psi,lpsi = fn.diff_matrix( grid_params_inv , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
    inv_psi = np.linalg.inv( dzz_psi )



    # first, check dzz_psi:
    psi_zzC = np.dot( dzz_psi , psi )
    psi_zzC[0] = psi_zzC[0] + lpsi*(z[0]**2.*zeta_wall) # sets psi_wall=d/dz(psi_wall)=0 to 2nd order accuracy
    psi_zzE = abs(psi_zz-psi_zzC)#/abs(psi_zz)*100.
    L_psi_zz[n] = np.amax(psi_zzE)

    plotname = figure_path + '/computed_solutions/psi_dzz_Nz%i.png' %(int(Nz))
    fig = plt.figure(figsize=(16,8)); 
    plt.subplot(1,2,1)
    plt.plot(psi_zz, z/H, color='red', linewidth=2, label=r"analytical")
    plt.plot(psi_zzC, z/H, linestyle='dashed',color='olive',linewidth=2, label=r"computed")
    #plt.plot(uzzDb, z/H, '--g', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{zz}\psi$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=1,fontsize=13); 
    #plt.ylim([-0.005,5.])
    #plt.axis([-0.005,1000.,-0.005,10.]) 
    #plt.axis([0.,20./H,0.,1./(2.*H)])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.subplot(1,2,2)
    plt.plot(psi_zz, z/H, color='red', linewidth=2, label=r"analytical")
    plt.plot(psi_zzC, z/H, linestyle='dashed',color='olive',linewidth=2, label=r"computed")
    #plt.plot(uzzDb, z/H, '--g', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{zz}\psi$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=1,fontsize=13); 
    plt.ylim([-0.005,0.1])
    plt.savefig(plotname,format="png")
    plt.close(fig);

    plotname = figure_path + '/computed_solutions/psi_dzz_error_Nz%i.png' %(int(Nz))
    fig = plt.figure(figsize=(16,8)); 
    plt.subplot(1,1,1)
    #plt.plot(uzz, z/H, 'b', label=r"analytical")
    #plt.plot(uzzD, z/H, '--r', label=r"computed")
    plt.semilogx(psi_zzE,z/H,color='olive',linewidth=2)
    plt.xlabel(r"$\partial_{zz}\psi$ $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    #plt.legend(loc=7,fontsize=13); 
    #plt.ylim([-0.005,0.1])
    plt.savefig(plotname,format="png")
    plt.close(fig);



    # next, check dzz_zeta:
    zeta_zzC = np.dot( dzz_zeta , psi_zz )
    """
    lw, l0, l1, l2 = fn.fornberg_weights(z[0], np.append(0.,z[0:3]) ,2)[:,2]
    zeta_zzC[0] = lw*zeta_wall + l0*psi_zz[0] + l1*psi_zz[1] + l2*psi_zz[2] 
    L_zeta_zz[n] = np.amax(zeta_zzE)
    """ 
    zeta_zzC[0] = zeta_zzC[0] + lzeta*zeta_wall # forward finite difference to the wall
    zeta_zzE = abs(psi_zzzz-zeta_zzC) #/abs(psi_zzzz)*100.
    L_zeta_zz[n] = np.amax(zeta_zzE)

    plotname = figure_path + '/computed_solutions/zeta_dzz_Nz%i.png' %(int(Nz))
    fig = plt.figure(figsize=(16,8)); 
    plt.subplot(1,2,1)
    plt.plot(psi_zzzz, z/H, color='red', linewidth=2, label=r"analytical")
    plt.plot(zeta_zzC, z/H, linestyle='dashed',color='olive',linewidth=2, label=r"computed")
    #plt.plot(uzzDb, z/H, '--g', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{zz}\zeta$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=1,fontsize=13); 
    #plt.ylim([-0.005,5.])
    #plt.axis([-0.005,1000.,-0.005,10.]) 
    #plt.axis([0.,20./H,0.,1./(2.*H)])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.subplot(1,2,2)
    plt.plot(psi_zzzz, z/H, color='red', linewidth=2, label=r"analytical")
    plt.plot(zeta_zzC, z/H, linestyle='dashed',color='olive',linewidth=2, label=r"computed")
    #plt.plot(uzzDb, z/H, '--g', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{zz}\zeta$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=1,fontsize=13); 
    plt.ylim([-0.005,0.1])
    plt.savefig(plotname,format="png")
    plt.close(fig);

    # next check the inversion: go from zeta to psi (could be done by integrating zeta twice? With Chebyshev points that could be easy)
    psiC = np.matmul( inv_psi , psi_zz )
    psiE = abs(psi-psiC)#/abs(psi_zz)*100.
    L_psi[n] = np.amax(psiE)


plotname = figure_path + 'zeta_inv_convergence.png'
fig = plt.figure(figsize=(8,8))
plt.subplot(1,1,1)
plt.loglog(Nr,L_psi,color='olive',linewidth=2.,label=r"$\psi$")
#plt.loglog(Nr,Lb,'b',label=r"cosine grid")
plt.loglog(Nr,(L_psi[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
#plt.title(r"$\partial_{zz}\psi$ error",fontsize=13) #: Dirchlet top & open bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png")
plt.close(fig);


plotname = figure_path + 'zeta_wall_convergence.png'
fig = plt.figure(figsize=(8,8))
plt.subplot(1,1,1)
plt.loglog(Nr,L_zeta_wall,color='olive',linewidth=2.,label=r"Thom (1933)")
plt.loglog(Nr,L_zeta_wall2,color='goldenrod',linewidth=2.,label=r"6 point")
#plt.loglog(Nr,Lb,'b',label=r"cosine grid")
plt.loglog(Nr,(L_zeta_wall[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"Vorticity boundary condition error",fontsize=13) #: Dirchlet top & open bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png")
plt.close(fig);

plotname = figure_path + 'psi_zz_convergence.png'
fig = plt.figure(figsize=(8,8))
plt.subplot(1,1,1)
plt.loglog(Nr,L_psi_zz,color='olive',linewidth=2.,label=r"$\partial_{zz}\psi$")
#plt.loglog(Nr,Lb,'b',label=r"cosine grid")
plt.loglog(Nr,(L_psi_zz[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
#plt.title(r"$\partial_{zz}\psi$ error",fontsize=13) #: Dirchlet top & open bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png")
plt.close(fig);

plotname = figure_path + 'zeta_zz_convergence.png'
fig = plt.figure(figsize=(8,8))
plt.subplot(1,1,1)
plt.loglog(Nr,L_zeta_zz,color='olive',linewidth=2.,label=r"$\partial_{zz}\zeta$")
#plt.loglog(Nr,Lb,'b',label=r"cosine grid")
plt.loglog(Nr,(L_zeta_zz[0]*0.5/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
#plt.title(r"$\partial_{zz}\psi$ error",fontsize=13) #: Dirchlet top & open bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png")
plt.close(fig);

"""

plotname = figure_path + 'grid_convergence.png'
fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.loglog(Nr,L1a,'r',label=r"neumann")
plt.loglog(Nr,L1b,'b',label=r"dirchlet")
#plt.loglog(Nr,Lb,'b',label=r"cosine grid")
plt.loglog(Nr,(L1a[0]*0.3/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"1st derivative of $\zeta$",fontsize=13) #: Dirchlet top & open bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.subplot(1,2,2)
plt.loglog(Nr,L2a,'r',label=r"neumann")
plt.loglog(Nr,L2b,'b',label=r"dirchlet")
#plt.loglog(Nr,Lb,'b',label=r"cosine grid")
plt.loglog(Nr,(L2a[0]*0.3/Nr[0]**(-2.))*Nr**(-2.),'k',label=r"$O(N^{-2})$")
plt.xlabel(r"$N$ grid points",fontsize=13)
plt.ylabel(r"L$_\infty$ error",fontsize=13)
plt.title(r"2nd derivative of $\zeta$",fontsize=13) #: Dirchlet top & open bottom BCs",fontsize=13)
plt.grid(); plt.legend(loc=1,fontsize=13)
plt.savefig(plotname,format="png")
plt.close(fig);


"""
