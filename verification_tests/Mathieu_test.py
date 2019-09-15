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

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})

figure_path = "./verification_tests/figures/Mathieu_test/"

#[0.974139101316803, 0.10452961672473293],
#[0.5850943481605508, 0.627177700348426],
#[-0.7435856826100729, 2.9268292682926855],
#[-0.22489705416534633, 2.1951219512195124],

mcurve = np.array([[-0.03246753246753187, 0.],
[-0.1290782388343361, 0.7317073170731732],
[-0.29073713742703244, 1.3588850174215992],
[-0.4849766957780899, 1.8815331010452923],
[-0.7116837865966783, 2.4041811846689853],
[-0.9060364722385623, 2.8222996515679384],
[-1.1005022851712738, 3.1358885017421585],
[-1.3597900357482233, 3.5540069686411115],
[-1.7810760667903525, 4.285714285714285],
[-2.494117380876963, 5.435540069686411],
[-3.077627946965926, 6.271777003484317],
[-4.568871894655867, 8.362369337979096],
[-1.9428480926738763, 4.808362369337978],
[-1.1972261188289055, 3.7630662020905845],
[0.2287433820534872, 1.3588850174215992],
[1., 0.],
[1.3645413819629848, 0.8362369337979061],
[1.6574279379157435, 1.463414634146332],
[1.9505407484501571, 2.2996515679442453],
[2.2112991538078646, 3.2404181184668985],
[2.4399294085705243, 4.494773519163758],
[2.5061088737046937, 5.644599303135891],
[2.4748857414362657, 6.79442508710801],
[2.31356622471605, 7.735191637630663],
[2.1847142404633697, 8.67595818815331],
[1.8612833159871496, 9.825783972125436],
[1.4083216435132826, 11.289198606271782],
[0.6313634101090546, 13.379790940766547],
[-0.6964116023349467, 16.515679442508713],
[-2.0894610615864977, 19.33797909407666],
[-4.2927281777456, 23.519163763066196],
[-0.015950948006695498, 15.261324041811847],
[0.89087741526766, 13.170731707317074],
[1.6031268383184756, 11.289198606271782],
[2.2829087289017593, 9.407665505226475],
[2.7683379338431617, 7.944250871080136],
[3.2855559075071294, 5.853658536585364],
[3.5115842345807504, 4.703832752613245],
[3.769853839540252, 3.3449477351916386],
[3.898366441920448, 2.0905923344947723],
[4., 0],
[4.092040363817368, 1.0452961672473862],
[4.385266301642609, 1.9860627177700323],
[4.710959771935382, 2.9268292682926855],
[5.1991040318566455, 3.9721254355400717],
[5.719942078827097, 5.226480836236931],
[6.240780125797549, 6.480836236933797],
[6.794311959817186, 7.944250871080136],
[7.087424770351602, 8.78048780487805],
[7.348183175709309, 9.721254355400696],
[7.7067966876329255, 11.080139372822295],
[7.935426942395585, 12.334494773519161],
[8.099461514095664, 13.902439024390247],
[8.101271550748907, 15.57491289198606],
[8.07038780035296, 17.038327526132406],
[7.909407665505226, 18.292682926829272],
[7.781008190415857, 19.65156794425087],
[7.619801800986471, 20.696864111498257],
[7.393547219331193, 21.637630662020904],
[7.10281008190416, 22.99651567944251],
[6.68254219647948, 24.66898954703833],
[6.100276030589619, 26.65505226480837],
[5.032467532467532, 30],
[3.9643196524729607, 33.03135888501742],
[2.6045296167247383, 36.585365853658544],
[4.4178469614009686, 32.09059233449477],
[5.485881714104712, 28.954703832752614],
[6.359563781166569, 26.23693379790941],
[7.006425630119011, 23.937282229965156],
[7.426806642834517, 22.369337979094077],
[8.008620299561068, 19.965156794425084],
[8.428435675822437, 17.87456445993032],
[8.815104755871305, 15.156794425087107],
[9.04022806461831, 13.170731707317074],
[9.199737544685279, 10.557491289198609],
[9.262523191094617, 8.57142857142857],
[9.260713154441374, 6.898954703832757],
[9.226435585320603, 5.226480836236931],
[9.12790171500973, 4.181184668989545],
[9.061948504457218, 3.2404181184668985],
[8.995203402868912, 1.5679442508710792],
[8.993619620797324, 0.10452961672473293],
[9.093284764016472, 2.1951219512195124],
[9.191705507036518, 3.1358885017421585],
[9.452690166975882, 4.285714285714285],
[9.84388433865786, 5.749128919860624],
[10.42999683243586, 7.31707317073171],
[11.211367030182364, 9.303135888501743],
[11.99251097334721, 11.080139372822295],
[12.740848002172042, 12.543554006968634],
[13.684895244128695, 14.843205574912893],
[14.628942486085341, 17.142857142857146],
[15.182813701977464, 18.919860627177698],
[15.606475406126977, 20.383275261324037],
[16.030702746730626, 22.369337979094077],
[16.324946830173317, 24.25087108013937],
[16.521901443504227, 26.23693379790941],
[16.588533417801713, 27.804878048780488],
[16.590343454454953, 29.477351916376307],
[16.527444680754783, 31.3588850174216],
[16.43151273813295, 32.717770034843205],
[16.33535454092946, 33.86759581881533],
[16.142359382777506, 35.54006968641115],
[15.884316032399658, 37.10801393728224],
[15.561790126250058, 39.09407665505226],
[14.302004615593466, 45.05226480836237],
[15.788723471650302, 38.78048780487805],
[16.17595818815331, 36.585365853658544],
[16.466129689126205, 34.70383275261324],
[16.82044436399837, 32.09059233449477],
[17.14161274265804, 28.85017421602788],
[17.302027241051633, 27.07317073170732],
[17.396941038056028, 24.773519163763062],
[17.394565364948647, 22.57839721254355],
[17.424770351599626, 20.487804878048784],
[17.259038870537125, 17.35191637630662],
[17.093986153219603, 14.843205574912893],
[16.92938594506539, 12.752613240418121],
[16.667043757636094, 10.348432055749122],
[16.50266980406353, 8.46689895470383],
[16.305828318023444, 6.585365853658537],
[16.141341237160056, 4.599303135888498],
[16.008982306891717, 2.2996515679442453],
[15.973912846735145, -0.10452961672473293],
[16.075048644735062, 3.3449477351916386],
[16.305036426987648, 5.853658536585364],
[16.66444182994706, 7.944250871080136],
[17.088782297841533, 10.034843205574916],
[17.544911534458578, 11.498257839721255],
[18.09866962305987, 13.170731707317074],
[18.652201457079506, 14.634146341463413],
[19.30302276121092, 15.993031358885013],
[19.986537852391514, 17.5609756097561]])




# =============================================================================

T = 2.*np.pi # radians, non-dimensional period
# dt in RK4 needs to be non-dimensional, as in dt = omg*T/Nt and omg*T = 2*pi

# undamped Hill equation coefficients: f(t) = a + b*cos(t), A(t) = [[0,1],[-f(t),0]]
Ngrid = 120 #400
a = np.linspace(-1.,5.,num=Ngrid,endpoint=True)
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
    #PhinOPH = np.eye(int(2),int(2),0,dtype=complex)

    PhinH,final_timeM = fn.rk4_time_step( paramsH, PhinH , T/100, T , 'Hills_equation' )
    #PhinOPH,final_timeOPM = fn.op_time_step( paramsH , PhinOPH , T/100, T , 'Hills_equation' )
  
    """
    TrH = np.abs(np.trace(PhinH))
    if TrH < 2.:
      strutt1[j,i] = 1. # 1 for stability
    
    TrOPH = np.abs(np.trace(PhinOPH))
    if TrOPH < 2.:
      strutt2[j,i] = 1.
    """
    modH = np.abs(np.linalg.eigvals(PhinH)) # eigenvals = floquet multipliers
    #if modH[0] < 1. and modH[1] < 1.:
    #  strutt3[j,i] = 1.
    strutt3[j,i] = np.log10(np.amax(modH))
    """
    modOPH = np.abs(np.linalg.eigvals(PhinOPH)) # eigenvals = floquet multipliers
    if modOPH[0] < 1. and modOPH[1] < 1.:
      strutt4[j,i] = 1.
    """

    # 
    """
    C = 1.
 
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
    """

print('...Mathieu equation test complete!\nInspect output plots in /figures/Mathieu_test to determine \n if Mathieu equation stability was computed properly\n') 

A,B = np.meshgrid(a,b)

plotname = figure_path +'strutt_eig_rk4.png' 
plottitle = r"$\mathrm{Mathieu}$ $\mathrm{equation}$, $\mathrm{log}_{10}\hspace{0.5mm}(\mu)$" # $\mathrm{Floquet}$ $\mathrm{stability}$" 
fig = plt.figure()
plt.subplot(111)
CS = plt.contourf(A,B,strutt3,100,cmap='gist_gray')
plt.colorbar(CS); 
plt.plot(mcurve[:,0]/4,mcurve[:,1]/4,color='goldenrod',linewidth=2.5)
plt.xlabel(r"$\delta$",fontsize=18);
plt.ylabel(r"$\varepsilon$",fontsize=18); 
plt.axis([-1.,5.,0.,8.])
plt.title(plottitle,fontsize=16);
plt.subplots_adjust(top=0.925, bottom=0.125, left=0.095, right=0.98, hspace=0.08, wspace=0.2)
plt.savefig(plotname,format="png"); plt.close(fig);

"""
plotname = figure_path +'strutt_Tr_rk4.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt1,cmap='gist_gray')
plt.plot(mcurve[:,0]/4,mcurve[:,1]/4,color='goldenrod',linewidth=3,label=r"$\mathrm{Mathieu}$ $\mathrm{equation}$")
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



plotname = figure_path +'strutt_eig_op.png' 
plottitle = r"Mathieu equation stablity" 
fig = plt.figure()
CS = plt.contourf(A,B,strutt4,cmap='gist_gray')
plt.xlabel(r"$\delta$",fontsize=13);
plt.ylabel(r"$\varepsilon$",fontsize=13); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);
"""
#


"""
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
"""

