

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Re = 1450
k = 0.35
C = 0.25
stat_path = './output/'
#C = 0.25
#stat_path = './output_C25_Re1400/'

def find_seed_zloc( arr , z , max_mode ):
    max_arr = np.amax(arr[:,0,max_mode])
    maxLoc = np.where(arr[:,0,max_mode] == max_arr)
    max_z = z[maxLoc[0][0]] 
    return max_z

def max_disturbance( Zeta , B , z , Z , N ):
    Zm = abs(np.real(Zeta))
    Bm = abs(np.real(B))
    Ni = np.shape(Zm)[1]
    max_Zm = np.amax(Zm[:,Ni-1,:])
    max_Bm = np.amax(Bm[:,Ni-1,:])
    max_arr = max(max_Zm,max_Bm)

    if max_arr == max_Zm:
        #print('Zeta maximum global instability')
        maxLoc = np.where(Zm == max_arr)
        max_z = z[maxLoc[0][0]] 
        max_mode = maxLoc[2][0]
        print(max_mode)
        if max_mode >= 200:
            max_z0 = find_seed_zloc( Bm , z , max_mode )
            print('\nOptimal: vorticity from a buoyancy disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $\zeta$ $\mathrm{from}$ $b$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,k,max_z0)  
        else:
            max_z0 = find_seed_zloc( Zm , z , max_mode )
            print('\nOptimal: vorticity from a vorticity disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $\zeta$ $\mathrm{from}$ $\zeta$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,k,max_z0)  

    if max_arr == max_Bm:
        print('Buoyancy maximum global instability')
        maxLoc = np.where(Bm == max_arr)
        max_z = z[maxLoc[0][0]]
        max_mode = maxLoc[2][0]
        print(max_z,max_mode)
        if max_mode <= 199:
            max_z0 = find_seed_zloc( Zm , z , max_mode )
            print('\nOptimal: vorticity from a vorticity disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $b$ $\mathrm{from}$ $\zeta$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,k,max_z0)
        else:
            max_z0 = find_seed_zloc( Bm , z , max_mode )
            print('\nOptimal: buoyancy from a buoyancy disturbance\n')
            instability_flag = r"$\mathrm{Re}=%i$, $\mathrm{C}=%.2f$, $k=%.2f$, $\mathrm{optimal}$ $\mathrm{growth}$ $\mathrm{occurs}$ $\mathrm{in}$ $b$ $\mathrm{from}$ $b$ $\mathrm{disturbance}$ $\mathrm{at}$ $z/\delta=%.2f$" %(int(Re),C,k,max_z0) 


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

    plotname = figure_path +'/most_unstable_mode_Re%i_k%i_C%i.png' %(int(Re),int(k*1000),int(C*1000))
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

    return

def max_loc( arr , z ):
    # find the maximum final time of zeta or b, determine
    # which one, then find the mode, and z location.
    maxElement = np.amax(np.amax(np.amax(arr)))
    maxLoc = np.where(arr == maxElement)
    max_z = z[maxLoc[0][0]]
    max_mode = maxLoc[2][0]
    return maxElement,max_z,max_mode


plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})


figure_path = './figures/contours'

Nz = 200 # vertical resolution
Ni = 17 # number of time stamps
Nm = int(Nz*2) # number of modes

B = np.zeros([Nz,Ni,Nm])
Zeta = np.zeros([Nz,Ni,Nm])
Psi = np.zeros([Nz,Ni,Nm])
n = np.zeros([Ni])

ifreq = 1000

for i in range(0,Ni): # time steps
    h5_filename = stat_path + 'profiles_%i.h5' %(int(ifreq*i))
    data = h5py.File( h5_filename , 'r')
    B[:,i,:] = data['/Bnr']
    Zeta[:,i,:] = data['/Znr']
    Psi[:,i,:] = data['/Pnr']
    n[i] = data['/n'].value

h5_filename = stat_path + 'profiles_0.h5' 
z = data['/z']

Z,N = np.meshgrid(n/(Ni*ifreq),z)

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

max_disturbance( Zeta , B , z , Z , N )

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
