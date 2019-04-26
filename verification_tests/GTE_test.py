# Global trunction error test
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
import sys
sys.path.insert(0, '/path/to/application/app/folder')
import functions as fn

#figure_path = "./figures/"
figure_path = './verification_tests/figures/GTE_test/'

# =============================================================================

T = 1.0 # s, period
T2 = 2.*np.pi

# analytical test:
omg = 2.0*np.pi/T # rads/s
alpha = 0.5 + 0.*1j
beta = 0.1 + 0.*1j
params = {'omg': omg, 'alpha': alpha, 'beta':beta}

# Mathieu's equation:
a = 8.
b = 1.
paramsH = {'a': a, 'b': b}

# loop over temporal resolution:
M = [2,10,100,1000,10000]
Linf = np.zeros(np.shape(M))
LinfOP = np.zeros(np.shape(M))
Ldiff = np.zeros(np.shape(M))
dT = np.zeros(np.shape(M)); dT2 = np.zeros(np.shape(M));
for m in range(0,np.shape(M)[0]):

 Nt = int(M[m]) 
 dt = T/Nt
 dt2 = T2/Nt
 #print(Nt,dt)
 dT[m] = dt
 dT2[m] = dt2

 Phin = np.eye(int(2),int(2),0,dtype=complex)
 PhinOP = np.eye(int(2),int(2),0,dtype=complex)
 PhinH = np.eye(int(2),int(2),0,dtype=complex)
 PhinOPH = np.eye(int(2),int(2),0,dtype=complex)

 Phin,final_time = fn.rk4_time_step( params, Phin , dt, T , 'analytical_test' )
 PhinOP,final_timeOP = fn.op_time_step( params , PhinOP , dt, T , 'analytical_test' )
 PhinH,final_timeH = fn.rk4_time_step( paramsH, PhinH , dt2, T2 , 'Hills_equation' )
 PhinOPH,final_timeOPH = fn.op_time_step( paramsH , PhinOPH , dt2, T2 , 'Hills_equation' )

 # analytical solution:
 Phia = np.matrix([[np.exp(-alpha*(final_time)),0.],[0.,np.exp(-beta*(final_time))]],dtype=complex)
 wa = np.linalg.eigvals(Phia)
 # RK4 computed solution:
 w = np.linalg.eigvals(Phin) # eigenvals, eigenvecs | eigenvals = floquet multipliers
 # ordered product computed solution:
 wOP,vOP = np.linalg.eig(PhinOP) # eigenvals, eigenvecs | eigenvals = floquet multipliers
 percent_error_mu1 = abs((w[0]-wa[0]))/abs(wa[0])*100.
 percent_error_mu2 = abs((w[1]-wa[1]))/abs(wa[1])*100.
 Linf[m] = np.amax([percent_error_mu1,percent_error_mu2])
 percent_error_mu1OP = abs((wOP[0]-wa[0]))/abs(wa[0])*100.
 percent_error_mu2OP = abs((wOP[1]-wa[1]))/abs(wa[1])*100.
 LinfOP[m] = np.amax([percent_error_mu1OP,percent_error_mu2OP])

 wH = np.linalg.eigvals(PhinH) 
 wOPH = np.linalg.eigvals(PhinOPH) 
 percent_diff_1 = abs((wH[0]-wOPH[0]))/abs(1./2.*(wH[0]+wOPH[0]))*100.
 percent_diff_2 = abs((wH[1]-wOPH[1]))/abs(1./2.*(wH[1]+wOPH[1]))*100.
 Ldiff[m] = np.amax([percent_diff_1,percent_diff_2])

check_flag = 0
if Ldiff[3] < 7.5e-06:
    check_flag = check_flag + 1

if Linf[3] < 9.9e-14:
    check_flag = check_flag + 1

if check_flag == 2:
     print('\n :) Global temporal discretization error sufficient \n for Mathieu equation & simple example\n') 
else:
     print('\n ERROR: Global temporal discretization not sufficient \n for Mathieu equation & simple example\n') 


plotname = figure_path +'GTE_loglog.png' 
plottitle = r"Floquet multiplier % GTE, for $\alpha=0.5,\beta=0.1$" 
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

plotname = figure_path +'GTE_loglog_diff.png' 
plottitle = r"Floquet multiplier global % difference GTE, Mathieu" 
fig = plt.figure()
plt.loglog(M,Ldiff,'b',label=r"RK4")
plt.loglog(M,(dT2**4.),'k',label=r"O$(dt^4)$")
plt.xlabel(r"$N$ time steps per period",fontsize=13);
plt.ylabel(r"$|error|_\infty$",fontsize=13); 
plt.legend(loc=1,fontsize=14); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);

"""
plotname = figure_path +'GTE_loglog_dt.png' 
plottitle = r"Floquet multiplier % GTE, for $\alpha=0.5,\beta=0.1$" 
fig = plt.figure()
plt.loglog(dT/T,Linf,'b',label=r"RK4")
plt.loglog(dT/T,LinfOP,'r',label=r"OP")
plt.loglog(dT/T,(dT**4.),'k',label=r"O$(dt^4)$")
plt.xlabel(r"$\Delta{t}/T$ normalized time step size",fontsize=13);
plt.ylabel(r"$|error|_\infty$",fontsize=13); 
plt.legend(loc=2,fontsize=14); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'GTE_semilogy.png' #%(start_time,end_time)
plottitle = r"Floquet multiplier % GTE, for $\alpha=0.5,\beta=0.1$" #, $\tau_w/U_0^2$ Re=%.2f, Pr=%.1f" #%(Re,Pr)
fig = plt.figure()
plt.semilogy(dT/T,Linf,'b',label=r"RK4")
plt.semilogy(dT/T,LinfOP,'r',label=r"OP")
plt.xlabel(r"$\Delta{t}/T$ normalized time step",fontsize=13); 
plt.ylabel(r"$|E|_\infty$",fontsize=13); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);
"""
