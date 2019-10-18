

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functions as fn

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})

figure_path = './figures/contours'
stat_path = './output/'

Re = 200
a = 0.3 # could be k or l
C = 0.25
vorticity_flag = 'streamwise'

Nz = 200 # vertical resolution
Ni = 10 # number of time stamps (no. profile.h5 files)


# =============================================================================

def find_seed_zloc( arr , z , max_mode ):
    max_arr = np.amax(arr[:,0,max_mode])
    maxLoc = np.where(arr[:,0,max_mode] == max_arr)
    max_z = z[maxLoc[0][0]] 
    return max_z

def max_disturbance( Zeta , B , z , Z , N , params ):

    # MAKE ZETA AND B IMAGINARY!

    Zm = abs(np.real(Zeta))
    Bm = abs(np.real(B))
    Ni = np.shape(Zm)[1]
    max_Zm = np.amax(Zm[:,Ni-1,:])
    max_Bm = np.amax(Bm[:,Ni-1,:])
    max_arr = max(max_Zm,max_Bm)

    dflag = 0

    if max_arr == max_Zm:
        #print('Zeta maximum global instability')
        maxLoc = np.where(Zm == max_arr)
        max_z = z[maxLoc[0][0]] 
        max_mode = maxLoc[2][0]
        print(max_mode)
        if max_mode >= 200:
            max_z0 = find_seed_zloc( Bm , z , max_mode )
            print('\nOptimal: vorticity from a buoyancy disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $\zeta$ $\mathrm{from}$ $b$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,a,max_z0)  
            dflag = 1
        else:
            max_z0 = find_seed_zloc( Zm , z , max_mode )
            print('\nOptimal: vorticity from a vorticity disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $\zeta$ $\mathrm{from}$ $\zeta$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,a,max_z0)  
            dflag = 1

    if max_arr == max_Bm:
        print('Buoyancy maximum global instability')
        maxLoc = np.where(Bm == max_arr)
        max_z = z[maxLoc[0][0]]
        max_mode = maxLoc[2][0]
        print(max_z,max_mode)
        if max_mode <= 199:
            max_z0 = find_seed_zloc( Zm , z , max_mode )
            print('\nOptimal: vorticity from a vorticity disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $b$ $\mathrm{from}$ $\zeta$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,a,max_z0)
            dflag = 2
        else:
            max_z0 = find_seed_zloc( Bm , z , max_mode )
            print('\nOptimal: buoyancy from a buoyancy disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $b$ $\mathrm{from}$ $b$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,a,max_z0) 
            dflag = 2

    """
    plotname = figure_path +'/checkT.png'
    fig = plt.figure(figsize=(10, 7))
    ax=plt.subplot(1,1,1)
    plt.plot(np.real(Zeta[:,Ni-1,max_mode]),z,'b')
    plt.plot(np.real(B[:,Ni-1,max_mode]),z,'r')
    plt.title(r"$\mathrm{mode}$ $%i$, ${t}=\mathrm{T}_{\mathrm{final}}$" %(max_mode))
    plt.savefig(plotname,format="png"); plt.close(fig);

    plotname = figure_path +'/check0.png'
    fig = plt.figure(figsize=(10, 7))
    ax=plt.subplot(1,1,1)
    plt.plot(np.real(Zeta[:,0,max_mode]),z,'b')
    plt.plot(np.real(B[:,0,max_mode]),z,'r')
    plt.title(r"$\mathrm{mode}$ $%i$, ${t}=0$" %(max_mode))
    plt.savefig(plotname,format="png"); plt.close(fig);
    """

    plotname = figure_path +'/most_unstable_mode_Re%i_k%i_C%i.png' %(int(Re),int(a*1000),int(C*1000))
    #plotname = figure_path +'/zeta/mode_%i.png' %(int(m)) 
    fig = plt.figure(figsize=(15, 7))
    ax=plt.subplot(1,2,1)
    plottitle = r"$\zeta$" #, $\mathrm{mode}$ $%i$" %(int(max_mode)) 
    CS = plt.contourf(Z,N,np.real(Zeta[:,:,max_mode]),100,cmap='gist_gray') # 
    #CP3=ax.contour(A18b/1000.,RE18b,np.log10(MUR18b),np.array([0.]),colors='dodgerblue',linewidths=3)
    #plt.xlabel(r"$\tau$",fontsize=20); 
    plt.ylabel(r"${z}/\delta$",fontsize=20); 
    plt.ylim([0.1,20.])
    plt.yticks([5.,10.,15.,20.])
    ax.set_yticklabels([r"$5$",r"$10$",r"$15$",r"$20$"], fontsize=18)
    plt.xticks([0.,0.5,0.99])
    ax.set_xticklabels([r"$0$",r"$\pi$",r"$2\pi$"], fontsize=18)
    cbar = plt.colorbar(CS);
    plt.title(plottitle,fontsize=20);

    ax=plt.subplot(1,2,2)
    plottitle = r"$b$" #, $\mathrm{mode}$ $%i$" %(int(max_mode)) 
    CS = plt.contourf(Z,N,np.real(B[:,:,max_mode]),100,cmap='gist_gray') # 
    #CP3=ax.contour(A18b/1000.,RE18b,np.log10(MUR18b),np.array([0.]),colors='dodgerblue',linewidths=3)
    #plt.xlabel(r"$\tau$",fontsize=20); 
    #plt.ylabel(r"${z}/\delta$",fontsize=20); 
    plt.ylim([0.1,20.])
    plt.yticks([5.,10.,15.,20.])
    ax.set_yticklabels([r"$5$",r"$10$",r"$15$",r"$20$"], fontsize=18)
    plt.xticks([0.,0.5,0.99])
    ax.set_xticklabels([r"$0$",r"$\pi$",r"$2\pi$"], fontsize=18)
    cbar = plt.colorbar(CS);
    plt.title(plottitle,fontsize=20);

    fig.suptitle(instability_flag,fontsize=20)
    plt.subplots_adjust(top=0.875, bottom=0.075, left=0.05, right=0.985, hspace=0.28, wspace=0.05)
    plt.savefig(plotname,format="png"); plt.close(fig);


    # if spanwise vorticity:




    # if streamwise vorticity:
    if params['vorticity_flag'] == 'streamwise':

        Nj = np.shape(params['timej'])[0]
        diff_zeta = np.zeros([Nz,Nj])
        baro_zeta = np.zeros([Nz,Nj])
        diff_b = np.zeros([Nz,Nj])
        conv_b = np.zeros([Nz,Nj])
        #n = np.zeros([Ni])
         
        #print(params['C2'])
        #print(params['a'])
        #print(params['eye_matrix'])

        for j in range(0,Nj): 
            # non-dimensional solutions
            Bs, U, V, Bz, Uz, Vz = fn.rotating_solution( params, params['timej'][j], 1 )
            Bz = Bz * (params['dS']*params['omg']) / ( (params['N'])**2 * params['U'] )  # non-dimensional

            # dynamical operator chunk for zeta in zeta equation (top left A11):
            diff_zeta[:,j] = np.abs(np.dot( ( params['dzz_zeta'] - (params['a']**2.*params['eye_matrix']) ) / 2. , Zeta[:,j,max_mode] ))
            zeta_wall = ((np.matmul(params['inv_psi'],Zeta[:,j,max_mode]))[0])*3./((params['z'][0])**2.) - Zeta[0,j,max_mode]/2.  # Woods (1954)
            diff_zeta[0,j] = np.abs(diff_zeta[0,j] + (params['lBC'] * zeta_wall) / 2.)
            # diffusion of streamwise vorticity

            # dynamical operator chunk for buoyancy in zeta equation: (top right A12)  
            baro_zeta[:,j] = np.abs(np.dot( params['C2']*1j*params['a']*params['eye_matrix']*np.cos(params['tht'])  , B[:,j,max_mode] ))
            #baro_zeta[:,j] = np.real(np.dot( params['eye_matrix']*np.cos(params['tht'])*params['a']  , B[:,j,max_mode] ))
            # baroclinic disturbance vorticity 

            # dynamical operator chunk for zeta in buoyancy equation: (bottom left A21)
            conv_b[:,j] = np.abs(np.dot( - np.matmul(Bz*1j*params['a']*params['eye_matrix']*params['Re']/2.,params['inv_psi']) , Zeta[:,j,max_mode] ) )
            # advection of mean stratification by disturbances        

            # dynamical operator chunk for buoyancy in buoyancy equation: (bottom right A22)
            diff_b[:,j] = np.abs(np.dot( ( params['dzz_b'] - (params['a']**2.*params['eye_matrix']) ) / (2.*params['Pr']) , B[:,j,max_mode] ))


        dflag = 1
        if dflag == 1: # vorticity mode

            plotname = figure_path +'/most_unstable_mode_zeta_eqn_Re%i_l%i_C%i.png' %(int(Re),int(a*1000),int(C*1000))
            fig = plt.figure(figsize=(12, 5))
            ax=plt.subplot(1,2,1)
            plottitle = r"$\Big|\frac{1}{2}(\partial_{zz}-l^2)\zeta_1\Big|$" #, $\mathrm{mode}$ $%i$" %(int(max_mode)) 
            CS = plt.contourf(Z,N,diff_zeta,100,cmap='gist_gray') #  
            plt.ylabel(r"${z}/\delta$",fontsize=20); 
            plt.ylim([0.1,20.])
            plt.yticks([5.,10.,15.,20.])
            ax.set_yticklabels([r"$5$",r"$10$",r"$15$",r"$20$"], fontsize=18)
            plt.xticks([0.,0.5,0.99])
            ax.set_xticklabels([r"$0$",r"$\pi$",r"$2\pi$"], fontsize=18)
            cbar = plt.colorbar(CS);
            plt.title(plottitle,fontsize=20);
            ax=plt.subplot(1,2,2)
            plottitle = r"$\Big|\frac{ilN^2\cos\theta{}}{\omega^2}b\Big|$" #, $\mathrm{mode}$ $%i$" %(int(max_mode)) 
            CS = plt.contourf(Z,N,baro_zeta,100,cmap='gist_gray') #  np.real(tilt_zeta)
            plt.ylabel(r"${z}/\delta$",fontsize=20); 
            plt.ylim([0.1,20.])
            plt.yticks([5.,10.,15.,20.])
            ax.set_yticklabels([r"$5$",r"$10$",r"$15$",r"$20$"], fontsize=18)
            plt.xticks([0.,0.5,0.99])
            ax.set_xticklabels([r"$0$",r"$\pi$",r"$2\pi$"], fontsize=18)
            cbar = plt.colorbar(CS);
            plt.title(plottitle,fontsize=20);
            plt.subplots_adjust(top=0.875, bottom=0.075, left=0.05, right=0.985, hspace=0.28, wspace=0.05)
            plt.savefig(plotname,format="png"); plt.close(fig);

        dflag = 2
        if dflag == 2: # buoyancy mode

            plotname = figure_path +'/most_unstable_mode_b_eqn_Re%i_l%i_C%i.png' %(int(Re),int(a*1000),int(C*1000))
            fig = plt.figure(figsize=(12, 5))
            ax=plt.subplot(1,2,1)
            plottitle = r"$\Big|-\frac{il\mathrm{Re}\partial_{z}B}{2}\psi\Big|$" #, $\mathrm{mode}$ $%i$" %(int(max_mode)) 
            CS = plt.contourf(Z,N,conv_b,100,cmap='gist_gray') #  
            plt.ylabel(r"${z}/\delta$",fontsize=20); 
            plt.ylim([0.1,20.])
            plt.yticks([5.,10.,15.,20.])
            ax.set_yticklabels([r"$5$",r"$10$",r"$15$",r"$20$"], fontsize=18)
            plt.xticks([0.,0.5,0.99])
            ax.set_xticklabels([r"$0$",r"$\pi$",r"$2\pi$"], fontsize=18)
            cbar = plt.colorbar(CS);
            plt.title(plottitle,fontsize=20);
            ax=plt.subplot(1,2,2)
            plottitle = r"$\Big|\frac{1}{2\mathrm{Pr}}(\partial_{zz}-l^2)b\Big|$" #, $\mathrm{mode}$ $%i$" %(int(max_mode)) 
            CS = plt.contourf(Z,N,diff_b,100,cmap='gist_gray') #  np.log10(
            plt.ylabel(r"${z}/\delta$",fontsize=20); 
            plt.ylim([0.1,20.])
            plt.yticks([5.,10.,15.,20.])
            ax.set_yticklabels([r"$5$",r"$10$",r"$15$",r"$20$"], fontsize=18)
            plt.xticks([0.,0.5,0.99])
            ax.set_xticklabels([r"$0$",r"$\pi$",r"$2\pi$"], fontsize=18)
            cbar = plt.colorbar(CS);
            plt.title(plottitle,fontsize=20);
            plt.subplots_adjust(top=0.875, bottom=0.075, left=0.05, right=0.985, hspace=0.28, wspace=0.05)
            plt.savefig(plotname,format="png"); plt.close(fig);

    return

def max_loc( arr , z ):
    # find the maximum final time of zeta or b, determine
    # which one, then find the mode, and z location.
    maxElement = np.amax(np.amax(np.amax(arr)))
    maxLoc = np.where(arr == maxElement)
    max_z = z[maxLoc[0][0]]
    max_mode = maxLoc[2][0]
    return maxElement,max_z,max_mode


# =============================================================================


# abyss:
T = 2.*np.pi # s, period
omg = 2.*np.pi/44700. # rads/s
nu = 1e-6
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
N = 1e-3 # 1/s, buoyancy frequency
f = 0.
thtc= ma.asin(omg/N) # radians    
tht = C*thtc # radians
Pr = 1. # Prandtl number
kap = nu/Pr # m^2/s, thermometric diffusivity
wall_flag = 'moving'

U = Re * (nu/dS) 

# grid:
grid_flag = 'uniform' #'hybrid cosine' #'  'cosine' # # 
wall_BC_flag = 'BC'
plot_freq = 500
Nz = 200 # 
H = 32. # = Hd/dS, non-dimensional grid height
CFL = 2. # 
Hd = H*dS # m, dimensional domain height (arbitrary choice)
z,dz = fn.grid_choice( grid_flag , Nz , H ) # non-dimensional grid

# modes:
Nm = int(Nz*2) # number of modes
B = np.zeros([Nz,Ni,Nm],dtype=complex)
Zeta = np.zeros([Nz,Ni,Nm],dtype=complex)
#Psi = np.zeros([Nz,Ni,Nm])
n = np.zeros([Ni])

ifreq = 500

for i in range(0,Ni): # time steps
    h5_filename = stat_path + 'profiles_%i.h5' %(int(ifreq*i))
    data = h5py.File( h5_filename , 'r')
    B[:,i,:] = data['/Bnr'] + data['/Bni']
    Zeta[:,i,:] = data['/Znr'] + data['/Zni']
    #Psi[:,i,:] = data['/Pnr']
    n[i] = data['/n'].value
    

#h5_filename = stat_path + 'profiles_0.h5' 
z = data['/z']

Z,N = np.meshgrid(n/(Ni*ifreq),z)
print(np.shape(N))
# find the most unstable of Zeta,B


#print(np.argmax(Zeta,axis=1))
#print(np.argmax(B,axis=1))

"""
# Get the maximum element from a Numpy array
max_Zeta,max_z,max_mode = max_loc( Zeta , z ) 
print('max. Zeta: ',max_Zeta)
print('z location of max. zeta: ',max_z)
print('mode of max. zeta: ',max_mode)

max_B,max_z,max_mode = max_loc( B , z ) 
print('max. buoyancy: ',max_B)
print('z location of max. buoyancy: ',max_z)
print('mode of max. buoyancy: ',max_mode)
"""

# pre-constructed matrices:
grid_params_dzz = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
grid_params_inv = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':wall_BC_flag} 
grid_params_b = {'H':H, 'Hd':Hd,'z':z,'dz':dz,'Nz':Nz, 'wall_BC_flag':' '} 
eye_matrix = np.eye( Nz , Nz , 0 , dtype=complex ) # identity matrix
dzz_zeta,lBC = fn.diff_matrix( grid_params_dzz , 'dirchlet 2' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
dzz_psi,lBC2 = fn.diff_matrix( grid_params_inv , 'dirchlet' , 'dirchlet' , diff_order=2 , stencil_size=3 ) 
dzz_b = fn.diff_matrix( grid_params_b , 'neumann' , 'neumann' , diff_order=2 , stencil_size=3 ) 
dz_b = fn.diff_matrix( grid_params_b , 'neumann' , 'neumann' , diff_order=1 , stencil_size=3 ) 

dzz_zeta = np.multiply(dzz_zeta,np.ones(np.shape(dzz_zeta)),dtype=complex) 
dzz_psi = np.multiply(dzz_psi,np.ones(np.shape(dzz_psi)),dtype=complex)
dzz_b = np.multiply(dzz_zeta,np.ones(np.shape(dzz_b)),dtype=complex) 
dz_b = np.multiply(dzz_zeta,np.ones(np.shape(dz_b)),dtype=complex)

inv_psi = np.linalg.inv( dzz_psi - (a**2.*eye_matrix) ) 
inv_psi = np.multiply(inv_psi,np.ones(np.shape(inv_psi)),dtype=complex)

timej = (n/(Ni*ifreq))*2.*np.pi
print(timej)

params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U, 'inv_psi':inv_psi, 'plot_freq':plot_freq, 'grid_flag':grid_flag,
          'Nz':Nz, 'Re':Re,'a':a, 'H':H, 'Hd':Hd, 'dzz_zeta':dzz_zeta, 'CFL':CFL, 'vorticity_flag':vorticity_flag,
          'Pr':Pr, 'tht':tht, 'N':1e-3, 'f':f, 'kap':kap, 'L':U/omg, 'wall_flag':wall_flag, 
          'dzz_b':dzz_b, 'dz_b':dz_b, 'C2':((1e-3/omg)**2.), 'inv_psi':inv_psi, 'timej':timej,
          'dS':dS, 'z':z, 'dz':dz, 'eye_matrix':eye_matrix, 'lBC':lBC, 'lBC2':lBC2} 




max_disturbance( Zeta , B , z , Z , N , params )

"""


for m in range(0,Nm): # modes
    #print(np.shape(N),np.shape(B[:,:,m]))

    plotname = figure_path +'/zeta/mode_%i.png' %(int(m)) 
    fig = plt.figure(figsize=(10, 7))
    ax=plt.subplot(1,1,1)
    plottitle = r"$\zeta$, $\mathrm{mode}$ $%i$" %(int(m)) 
    CS = plt.contourf(Z,N,Zeta[:,:,m],100,cmap='gist_gray') # 
    #CP3=ax.contour(A18b/1000.,RE18b,np.log10(MUR18b),np.array([0.]),colors='dodgerblue',linewidths=3)
    plt.xlabel(r"$t/(2\pi)$",fontsize=20); 
    plt.ylabel(r"${z}/\delta$",fontsize=20); 
    #plt.ylim([0.,150.])
    #plt.yticks([600.,1000.,1400.,1800.,2200.,2600.] )
    #plt.xticks([0.15,0.35,0.55,0.75,0.95,1.15,1.35] )
    cbar = plt.colorbar(CS);
    #cbar.add_lines(CP3)  
    #plt.legend(loc=3,fontsize=16,facecolor='white', framealpha=1)
    plt.title(plottitle,fontsize=17);
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.98, hspace=0.28, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);


    plotname = figure_path +'/b/mode_%i.png' %(int(m)) 
    fig = plt.figure(figsize=(10, 7))
    ax=plt.subplot(1,1,1)
    plottitle = r"$b$, $\mathrm{mode}$ $%i$" %(int(m)) 
    CS = plt.contourf(Z,N,B[:,:,m],100,cmap='gist_gray') # 
    #CP3=ax.contour(A18b/1000.,RE18b,np.log10(MUR18b),np.array([0.]),colors='dodgerblue',linewidths=3)
    plt.xlabel(r"$t/(2\pi)$",fontsize=20); 
    plt.ylabel(r"${z}/\delta$",fontsize=20); 
    #plt.ylim([0.,150.])
    #plt.yticks([600.,1000.,1400.,1800.,2200.,2600.] )
    #plt.xticks([0.15,0.35,0.55,0.75,0.95,1.15,1.35] )
    cbar = plt.colorbar(CS);
    #cbar.add_lines(CP3)  
    #plt.legend(loc=3,fontsize=16,facecolor='white', framealpha=1)
    plt.title(plottitle,fontsize=17);
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.98, hspace=0.28, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);


    plotname = figure_path +'/psi/mode_%i.png' %(int(m)) 
    fig = plt.figure(figsize=(10, 7))
    ax=plt.subplot(1,1,1)
    plottitle = r"$\psi$, $\mathrm{mode}$ $%i$" %(int(m)) 
    CS = plt.contourf(Z,N,Psi[:,:,m],100,cmap='gist_gray') # 
    #CP3=ax.contour(A18b/1000.,RE18b,np.log10(MUR18b),np.array([0.]),colors='dodgerblue',linewidths=3)
    plt.xlabel(r"$t/(2\pi)$",fontsize=20); 
    plt.ylabel(r"${z}/\delta$",fontsize=20); 
    #plt.ylim([0.,150.])
    #plt.yticks([600.,1000.,1400.,1800.,2200.,2600.] )
    #plt.xticks([0.15,0.35,0.55,0.75,0.95,1.15,1.35] )
    cbar = plt.colorbar(CS);
    #cbar.add_lines(CP3)  
    #plt.legend(loc=3,fontsize=16,facecolor='white', framealpha=1)
    plt.title(plottitle,fontsize=17);
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.98, hspace=0.28, wspace=0.15)
    plt.savefig(plotname,format="png"); plt.close(fig);
"""



