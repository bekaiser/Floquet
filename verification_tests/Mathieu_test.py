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

figure_path = "./verification_tests/figures/Mathieu_test/"


# =============================================================================

T = 2.*np.pi # radians, non-dimensional period
# dt in RK4 needs to be non-dimensional, as in dt = omg*T/Nt and omg*T = 2*pi

# undamped Hill equation coefficients: f(t) = a + b*cos(t), A(t) = [[0,1],[-f(t),0]]
Ngrid = 10 #400
a = np.linspace(-2.,10.,num=Ngrid,endpoint=True)
b = np.linspace(0.,8.,num=Ngrid,endpoint=True)

strutt1 = np.zeros([Ngrid,Ngrid]); strutt2 = np.zeros([Ngrid,Ngrid])
strutt3 = np.zeros([Ngrid,Ngrid]); strutt4 = np.zeros([Ngrid,Ngrid])

strutt12 = np.zeros([Ngrid,Ngrid]); strutt22 = np.zeros([Ngrid,Ngrid])
strutt32 = np.zeros([Ngrid,Ngrid]); strutt42 = np.zeros([Ngrid,Ngrid])

count = 1

print('\nMathieu equation test running...\n')
for i in range(0,Ngrid):
  for j in range(0,Ngrid):
 
    #print(count)
    count = count + 1

    paramsH = {'a': a[i], 'b': b[j], 'freq':0} 

    #
  
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

    # 
  
    C = 1.e16
 
    PhinH2 = np.eye(int(2),int(2),0,dtype=complex) / C
    PhinOPH2 = np.eye(int(2),int(2),0,dtype=complex) / C

    PhinH2,final_timeM2 = fn.rk4_time_step( paramsH, PhinH2 , T/100, T , 'Hills_equation' )
    PhinOPH2,final_timeOPM2 = fn.op_time_step( paramsH , PhinOPH2 , T/100, T , 'Hills_equation' )

    TrH2 = np.abs(np.trace(PhinH2)) * C
    if TrH2 < 2.:
      strutt12[j,i] = 1. # 1 for stability
    
    TrOPH2 = np.abs(np.trace(PhinOPH2)) * C
    if TrOPH2 < 2.:
      strutt22[j,i] = 1.

    modH2 = np.abs(np.linalg.eigvals(PhinH2)) * C # eigenvals = floquet multipliers
    if modH2[0] < 1. and modH2[1] < 1.:
      strutt32[j,i] = 1.

    modOPH2 = np.abs(np.linalg.eigvals(PhinOPH2)) * C # eigenvals = floquet multipliers
    if modOPH2[0] < 1. and modOPH2[1] < 1.:
      strutt42[j,i] = 1.


print('...Mathieu equation test complete!\nInspect output plots in /figures/Mathieu_test to determine \n if Mathieu equation stability was computed properly\n') 

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

#

plotname = figure_path +'strutt_Tr_rk4_2.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt12,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'strutt_Tr_op_2.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt22,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'strutt_eig_rk4_2.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt32,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'strutt_eig_op_2.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt42,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);


