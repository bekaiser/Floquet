# Local trunction error test
# Bryan Kaiser
# 3/14/2019

# Note: see LaTeX document "floquet_primer" for analytical solution derivation

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import functions as fn

figure_path = "./figures/"


# =============================================================================

T = 1.0 # s, period

# analytical test:
omg = 2.0*np.pi/T # rads/s
alpha = 0.5 + 0.*1j
beta = 0.1 + 0.*1j
params = {'omg': omg, 'alpha': alpha, 'beta':beta}

# loop over temporal resolution:
M = [2,10,100,1000,10000]
Linf = np.zeros(np.shape(M))
LinfOP = np.zeros(np.shape(M))
dT = np.zeros(np.shape(M))
for m in range(0,np.shape(M)[0]):

 Nt = int(M[m]) 
 dt = T/Nt
 print(Nt,dt)
 dT[m] = dt

 Phin = np.eye(int(2),int(2),0,dtype=complex)
 PhinOP = np.eye(int(2),int(2),0,dtype=complex)

 Phin,final_time = fn.rk4_time_step( params, Phin , dt, dt , 'analytical_test' )
 PhinOP,final_timeOP = fn.op_time_step( params , PhinOP , dt, dt , 'analytical_test' )

 # analytical solution:
 Phia = np.matrix([[np.exp(-alpha*(final_time)),0.],[0.,np.exp(-beta*(final_time))]],dtype=complex)
 wa,va = np.linalg.eig(Phia)
 # RK4 computed solution:
 w,v = np.linalg.eig(Phin) # eigenvals, eigenvecs | eigenvals = floquet multipliers
 # ordered product computed solution:
 wOP,vOP = np.linalg.eig(PhinOP) # eigenvals, eigenvecs | eigenvals = floquet multipliers
 percent_error_mu1 = abs((w[0]-wa[0]))/abs(wa[0])*100.
 percent_error_mu2 = abs((w[1]-wa[1]))/abs(wa[1])*100.
 Linf[m] = np.amax([percent_error_mu1,percent_error_mu2])
 percent_error_mu1OP = abs((wOP[0]-wa[0]))/abs(wa[0])*100.
 percent_error_mu2OP = abs((wOP[1]-wa[1]))/abs(wa[1])*100.
 LinfOP[m] = np.amax([percent_error_mu1OP,percent_error_mu2OP])


plotname = figure_path +'LTE_loglog.png' 
plottitle = r"Floquet multiplier % LTE, for $\alpha=0.5,\beta=0.1$" 
fig = plt.figure()
plt.loglog(M,Linf,'b',label=r"RK4")
plt.loglog(M,LinfOP,'r',label=r"OP")
plt.loglog(M,(dT**4.),'k',label=r"O$(dt^4)$")
plt.xlabel(r"$N$ time steps per period",fontsize=13);
plt.ylabel(r"$|error|_\infty$",fontsize=13); 
plt.legend(loc=1,fontsize=14); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);






