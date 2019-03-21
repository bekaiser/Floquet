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

T = 2.*np.pi # radians, non-dimensional period
# dt in RK4 needs to be non-dimensional, as in dt = omg*T/Nt and omg*T = 2*pi

# undamped Hill equation coefficients: f(t) = a + b*cos(t), A(t) = [[0,1],[-f(t),0]]
Ngrid = 400
a = np.linspace(-2.,10.,num=Ngrid,endpoint=True)
b = np.linspace(0.,8.,num=Ngrid,endpoint=True)

strutt1 = np.zeros([Ngrid,Ngrid]); strutt2 = np.zeros([Ngrid,Ngrid])
strutt3 = np.zeros([Ngrid,Ngrid]); strutt4 = np.zeros([Ngrid,Ngrid])

count = 1

for i in range(0,Ngrid):
  for j in range(0,Ngrid):
 
    print(count)
    count = count + 1

    paramsH = {'a': a[i], 'b': b[j]} 
    PhinH = np.eye(int(2),int(2),0,dtype=complex)
    PhinOPH = np.eye(int(2),int(2),0,dtype=complex)
    PhinH,final_timeM = fn.rk4_time_step( paramsH, PhinH , T/100, T , 'Hills_equation' )
    PhinOPH,final_timeOPM = fn.op_time_step( paramsH , PhinOPH , T/100, T , 'Hills_equation' )

    TrH = np.abs(np.trace(PhinH))
    if TrH < 2.:
      strutt1[j,i] = 1. # 1 for stability
    
    TrOPH = np.abs(np.trace(PhinOPH))
    if TrOPH < 2.:
      strutt2[j,i] = 1.

    modH = np.abs(np.linalg.eigvals(PhinH)) # eigenvals = floquet multipliers
    if modH[0] < 1. and modH[1] < 1.:
      strutt3[j,i] = 1.

    modOPH = np.abs(np.linalg.eigvals(PhinOPH)) # eigenvals = floquet multipliers
    if modOPH[0] < 1. and modOPH[1] < 1.:
      strutt4[j,i] = 1.


A,B = np.meshgrid(a,b)

plotname = figure_path +'strutt_Tr_rk4.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt1,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'strutt_Tr_op.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt2,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'strutt_eig_rk4.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt3,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'strutt_eig_op.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt4,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);
