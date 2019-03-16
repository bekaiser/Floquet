# Stability calculation test
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

T = 2.*np.pi # s, period

# undamped Hill equation coefficients: f(t) = a + b*cos(t), A(t) = [[0,1],[-f(t),0]]
Ngrid = 50
k = np.logspace(-4.,1.,num=Ngrid,base=10.,endpoint=True) #np.linspace(1.,10.,num=Ngrid,endpoint=True)
c = np.logspace(-3.,1.,num=Ngrid,base=10.,endpoint=True)
#np.linspace(0.,1.,num=Ngrid,endpoint=False) # 


strutt = np.zeros([Ngrid,Ngrid]); #strutt2 = np.zeros([Ngrid,Ngrid])

count = 1

for i in range(0,Ngrid):
  for j in range(0,Ngrid):
 
    print(count)
    count = count + 1

    params = {'k': k[i], 'c2': c[j]**2.} 
    Phin = np.eye(int(2),int(2),0,dtype=complex)
    #PhinOPH = np.eye(int(2),int(2),0,dtype=complex)
    Phin,final_time = fn.rk4_time_step( params, Phin , T/100, T , 'inviscid_buoyancy' )
    #PhinOPH,final_timeOPM = fn.op_time_step( paramsH , PhinOPH , T/100, T , 'Hills_equation' )

    mod = np.abs(np.linalg.eigvals(Phin)) # eigenvals = floquet multipliers
    if mod[0] < 1. and mod[1] < 1.:
      strutt[j,i] = 1.
    """
    modOPH = np.abs(np.linalg.eigvals(PhinOPH)) # eigenvals = floquet multipliers
    if modOPH[0] < 1. and modOPH[1] < 1.:
      strutt4[j,i] = 1.
    """

A,B = np.meshgrid(np.log10(k),np.log10(c))

plotname = figure_path +'strutt_buoy_eig_rk4.png' 
plottitle = r"inviscid buoyancy equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt,cmap='gist_gray')
plt.xlabel(r"log$_{10}(k)$ non-dimensional wavenumber",fontsize=13);
plt.ylabel(r"log$_{10}$C",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

"""
plotname = figure_path +'strutt_eig_op.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt4,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);
"""
