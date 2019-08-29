# verifies first and second derivative Lagrange polynomial computation
# Bryan Kaiser

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/path/to/application/app/folder')
import functions as fn

figure_path = './verification_tests/figures/discretization_test/'

# try adding extra grid point to sims
# try changing the fornberg center point to z=0
# what if I just set psi[0] (first grid point psi to zero, as is true to second order, then specify other terms in an extended thing)

# =============================================================================
# loop over Nz resolution Chebyshev node grid

max_exp = 13 # power of two, must be equal to or greater than 5 (maximum N = 2^max)
Nr = np.power(np.ones([max_exp-3])*2.,np.linspace(4.,max_exp,max_exp-3)) # resolution 
Ng = int(np.shape(Nr)[0]) # number of resolutions to try
L1a = np.zeros([Ng])
L1b = np.zeros([Ng])
L2a = np.zeros([Ng])
L2b = np.zeros([Ng])
#print(Nr)

H = 32. # = Hd/dS, non-dimensional grid height
nu = 1e-6
omg = 2.*np.pi/44700.
dS = np.sqrt(2.*nu/omg)
Hd = H*dS 
wall_BC_off_flag = ' ' 



for n in range(0, Ng): 
      
    Nz = int(Nr[n]) # resolution
    #print('Number of grid points: ',Nz)
 
    dz = H/Nz
    z = np.linspace(dz, Nz*dz, num=Nz)-dz/2. # uniform, non-dimensional grid
    params = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_off_flag} 

    if n == 4:
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
    """
    u = np.cos(2.*np.pi/H*z)-np.ones(np.shape(z))
    uz = -2.*np.pi/H*np.sin(2.*np.pi/H*z)
    uzz = -np.cos(2.*np.pi/H*z)*(2.*np.pi/H)**2.

    if n == 4:
        plotname = figure_path + 'solutions.png'
        fig = plt.figure(figsize=(24,8)); 
        plt.subplot(1,3,1)
        plt.plot(u, z/H, 'b') #, label=r"u")
        #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
        plt.xlabel(r"$u$", fontsize=13)
        plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
        #plt.ylim([-0.001,0.05])
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        #plt.legend(loc=2,fontsize=13); 
        #plt.axis([0.,20./H,0.,1./(2.*H)])
        plt.subplot(1,3,2)
        plt.plot(uz, z/H, 'b') #, label=r"analytical")
        #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
        plt.xlabel(r"$\partial_{z}u$", fontsize=13)
        plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        plt.subplot(1,3,3)
        plt.plot(uzz, z/H, 'b') #, label=r"analytical")
        #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
        plt.xlabel(r"$\partial_{zz}u$", fontsize=13)
        plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
        plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
        #plt.legend(loc=2,fontsize=13); 
        plt.savefig(plotname,format="png")
        plt.close(fig);

    # derivatives:
    uzDa = np.dot( fn.diff_matrix( params , 'neumann' , 'neumann' , diff_order=1 , stencil_size=3 ) , u ) 
    #uzDb = np.dot( fn.partial_z( params , 'dirchlet' , 'neumann' ) , u ) 
    uzDb = np.dot( fn.diff_matrix( params , 'dirchlet' , 'neumann' , diff_order=1 , stencil_size=3 ) , u ) 
    uzzDa = np.dot( fn.diff_matrix( params , 'neumann' , 'neumann' , diff_order=2 , stencil_size=3 ) , u ) 
    uzzDb = np.dot( fn.diff_matrix( params , 'dirchlet' , 'neumann' , diff_order=2 , stencil_size=3 ) , u ) 

    plotname = figure_path + 'dz_Nz%i.png' %(int(Nz))
    fig = plt.figure(figsize=(16,8)); 
    plt.subplot(1,2,1)
    plt.plot(uz, z/H, 'k', label=r"analytical")
    plt.plot(uzDa, z/H, '--r', label=r"neumann")
    #plt.plot(uzDb, z/H, '--g', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{z}u$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=2,fontsize=13); 
    #plt.ylim([-0.005,5.])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.subplot(1,2,2)
    plt.plot(uz, z/H, 'k', label=r"analytical")
    #plt.plot(uzDa, z/H, '--r', label=r"neumann")
    plt.plot(uzDb, z/H, '--b', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{z}u$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13) #
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=2,fontsize=13); 
    #plt.axis([0.,20./H,0.,1./(2.*H)])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.savefig(plotname,format="png")
    plt.close(fig);


    plotname = figure_path + 'dzz_Nz%i.png' %(int(Nz))
    fig = plt.figure(figsize=(16,8)); 
    plt.subplot(1,2,1)
    plt.plot(uzz, z/H, 'k', label=r"analytical")
    plt.plot(uzzDa, z/H, '--r', label=r"neumann")
    #plt.plot(uzzDb, z/H, '--g', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{zz}u$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=2,fontsize=13); 
    #plt.ylim([-0.005,5.])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.subplot(1,2,2)
    plt.plot(uzz, z/H, 'k', label=r"analytical")
    #plt.plot(uzzDa, z/H, '--r', label=r"neumann")
    plt.plot(uzzDb, z/H, '--b', label=r"dirchlet")
    #plt.plot(abs(uzz-uzzD)/abs(uzz)*100.,z/H,'b')
    plt.xlabel(r"$\partial_{zz}u$", fontsize=13) # $\mathrm{percent}$ $\mathrm{error}$", fontsize=13) #
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=2,fontsize=13); 
    #plt.axis([0.,20./H,0.,1./(2.*H)])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.savefig(plotname,format="png")
    plt.close(fig);

    plotname = figure_path + 'dzz_error_Nz%i.png' %(int(Nz))
    fig = plt.figure(figsize=(16,8)); 
    plt.subplot(1,2,1)
    #plt.plot(uzz, z/H, 'b', label=r"analytical")
    #plt.plot(uzzD, z/H, '--r', label=r"computed")
    plt.plot(abs(uzz-uzzDa)/abs(uzz)*100.,z/H,'k',label="neumann")
    plt.xlabel(r"$\partial_{zz}u$ $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=7,fontsize=13); 
    plt.ylim([-0.005,0.1])
    plt.subplot(1,2,2)
    #plt.plot(uzz, z/H, 'b', label=r"analytical")
    #plt.plot(uzzD, z/H, '--r', label=r"computed")
    plt.plot(abs(uzz-uzzDb)/abs(uzz)*100.,z/H,'k',label="dirchlet")
    plt.xlabel(r"$\partial_{zz}u$ $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=7,fontsize=13); 
    plt.ylim([-0.005,0.1])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.savefig(plotname,format="png")
    plt.close(fig);

    plotname = figure_path + 'dz_error_Nz%i.png' %(int(Nz))
    fig = plt.figure(figsize=(16,8)); 
    plt.subplot(1,2,1)
    #plt.plot(uzz, z/H, 'b', label=r"analytical")
    #plt.plot(uzzD, z/H, '--r', label=r"computed")
    plt.plot(abs(uz-uzDa)/abs(uz)*100.,z/H,'k',label="neumann")
    plt.xlabel(r"$\partial_{z}u$ $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=7,fontsize=13); 
    plt.ylim([-0.005,0.1])
    plt.subplot(1,2,2)
    #plt.plot(uzz, z/H, 'b', label=r"analytical")
    #plt.plot(uzzD, z/H, '--r', label=r"computed")
    plt.plot(abs(uz-uzDb)/abs(uz)*100.,z/H,'k',label="dirchlet")
    plt.xlabel(r"$\partial_{z}u$ $\mathrm{percent}$ $\mathrm{error}$", fontsize=13)
    plt.ylabel(r"$z/H$",fontsize=13); plt.grid()
    plt.title(r"uniform grid N = %i" %(Nz),fontsize=13)
    plt.legend(loc=7,fontsize=13);
    plt.ylim([-0.005,0.1]) 
    #plt.ylim([-0.005,5.])
    #plt.axis([-0.005,1000.,-0.005,10.])
    plt.savefig(plotname,format="png")
    plt.close(fig);

    """
    per = abs(uzz-uzzDa)/abs(uzz)*100.
    #per[0:int(Nz/2)] = 0.
    #per[int(Nz/2):Nz] = 0.
    L1a[n] = np.max(per)
    """

    """
    L1a[n] = np.max(abs(uz-uzDa)/abs(uz)*100.)
    L1b[n] = np.max(abs(uz-uzDb)/abs(uz)*100.)
    L2a[n] = np.max(abs(uzz-uzzDa)/abs(uzz)*100.)
    L2b[n] = np.max(abs(uzz-uzzDb)/abs(uzz)*100.)
    """

    # bottom half only:
    L1a[n] = np.max(abs(uz[0:int(Nz/2)]-uzDa[0:int(Nz/2)])/abs(uz[0:int(Nz/2)])*100.)
    L1b[n] = np.max(abs(uz[0:int(Nz/2)]-uzDb[0:int(Nz/2)])/abs(uz[0:int(Nz/2)])*100.)
    L2a[n] = np.max(abs(uzz[0:int(Nz/2)]-uzzDa[0:int(Nz/2)])/abs(uzz[0:int(Nz/2)])*100.)
    L2b[n] = np.max(abs(uzz[0:int(Nz/2)]-uzzDb[0:int(Nz/2)])/abs(uzz[0:int(Nz/2)])*100.)




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



