# 

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chi2
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


# =============================================================================    
# classes and functions

class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
 
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
 
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
 
        return patch

def spectral_density( uz, t, dt, band_flag ):
  
  N = np.shape([t])[1] # unitless number of discrete samples
  L = t[N-1]-t[0]+dt # s, time series length

  # Hann window
  window = signal.hann(N)
  uz = np.multiply(uz,window)

  # wavenumbers/frequencies
  k = np.zeros([N])
  k[1:int(N/2)+1] = np.linspace( 1., N/2., num=int(N/2) )*(2.*np.pi/L) # rads/s
  k[int(N/2)+1:N] = -np.fliplr( [np.linspace(1., N/2.-1., num=int(N/2-1) )*(2.*np.pi/L)])[0]    

  # spectral density:
  UZ = np.fft.fft(uz-np.mean(uz))
  PHI = (np.conj(UZ)*UZ) # * 2.*L/N**2.
  PHI1 = PHI[0:int(N/2+1)]*2.
  k1 = k[0:int(N/2+1)]

  # band-averaging:
  if band_flag == 1: 
    nb = 2 # number of points per band
    Nb = int(N/(2*nb)) # number of bands
    PHI1b = np.zeros([Nb],dtype=complex)
    k1b = np.zeros([Nb])
    for j in range(0,Nb):
      #print(int(j*nb))
      #print(int((j+1)*nb-1))
      PHI1b[j] = np.mean(PHI1[int(j*nb):int((j+1)*nb-1)])
      k1b[j] = np.mean(k1[int(j*nb):int((j+1)*nb-1)])
  else:
    PHI1b = PHI1 
    k1b = k1
    Nb = N

  # 95% confidence intervals for chi-square distributed random error
  alpha = 0.05 # 95% confidence interval
  nu = 2*Nb-1 # number degrees of freedom
  [hi,lo] = chi2.interval(1-alpha,nu) # nu/Chi^2_(nu/alpha/2) & nu/Chi^2_(nu/(1-alpha/2))
  lo = nu/lo
  hi = nu/hi
  ub = PHI1b*hi # upper bound
  lb = PHI1b*lo # lower bound

  return PHI1b,ub,lb,k1b # spectral density, confidence interval upper/lower bounds, frequencies


def find_interval( t ):
  # returns the indices of the beginning of the first complete tidal cycle 
  # and the last complete tidal cycle  
  N = np.shape([t])[1]
  ti = ma.ceil(t[0]) # the beginning of the first complete tidal cycle
  tf = ma.floor(t[N-1]) # the last complete tidal cycle
  ni = np.zeros([N]) # index for the beginning
  j = 0 # counter for the length of the set of complete cycles
  for m in range(0,N):
    if t[m] >= ti:
      ni[m] = 0
    else: 
      ni[m] = 1
    if t[m] >= ti and t[m] <= tf:
      j = j + 1
  Nt = make_even(int(j)) # index length of the set of complete tidal cycles 
  Nt0 = int(sum(ni)) # initial index of the set of complete tidal cycles
  return Nt,Nt0

def make_even( N ):
  # is the integer N odd? If so, returns N-1
  if N % 2 == 0:
    return N
  else:
    N = N-1
    return N

def time_interval( t0 ):
  # t must be the unitless form, t/T
  # dt is dimensional
  [Nt,Nt0] = find_interval( t0 )
  t = np.zeros([Nt])
  t = t0[int(Nt0):int(Nt0+Nt)] # unitless
  Ncycles = int(round(t[Nt-1]-t[0]))
  t = t*T # s
  #L = t[Nt-1]-t[0]+dt # s
  #u = u0[int(Nt0):int(Nt0+Nt)] 
  return Nt,Nt0,Ncycles,t

def parseval_check( u , UZD , L ):
  var = np.mean(np.power(u-np.mean(u),2.))
  fcf = sum(UZD/L) # Fourier coefficients of spectral density
  #UZM = np.fft.fft(uzm7-np.mean(uzm7))
  #fcf = sum(np.power(abs(UZM/np.shape(UZM)[0]),2.))
  string = 'Parseval theorem:\nVariance - summed Fourier coefficients = %.16f' %(abs(var-fcf))
  print(string)
  return

def spectral_plot( PHI, UB, LB, k, L, plottitle, axes, colors, color_mean, color_shading, legend_name, figure_path, plot_name ): 
  plt.fill_between(k,np.real(UB)/L,np.real(LB)/L,color=color_shading,alpha=0.5) 
  p1=plt.loglog(k,np.real(PHI)/L, color=color_mean)
  #p2=plt.loglog(k[20:int(np.shape(k1b7)[0]-10500)]*L7/(2.*np.pi*Ncycles7)*5.,3e-10*np.power(k1b7[20:int(np.shape(k1b7)  [0]-10500)]*L7/200.,-10./3.), '--k')
  plt.xlabel('cycles per tidal oscillation')
  #plottitle = 'spectral density of du/dz' #, forcing k =  %.5f rads/m' %(kx)
  plt.title(plottitle)
  plt.grid()
  plt.axis(axes) # axes = [2e-2,4e2,1e-23,2e-4]
  bg = np.array([1,1,1])  # background of the legend is white
  #colors = ['green'] #,'blue'] #,'green','green']
  # with alpha = .5, the faded color is the average of the background and color
  colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
  plt.legend([0], legend_name,handler_map={0: LegendObject(colors[0], colors_faded[0])},loc=1)
               # 1: LegendObject(colors[1], colors_faded[1])},loc=1)
               #2: LegendObject(colors[2], colors_faded[2], dashed=True),    
  plt.savefig( figure_path + plot_name ) # plotname = '/uz_spectral_density_loglog_7.png'
  plt.close()
  return

def spectral_subplot( PHI, UB, LB, k, L, plottitle, axes, colors, color_mean, color_shading, legend_name, figure_ylabel, ytick_label_flag , xtick_label_flag): 
  plt.fill_between(k,np.real(UB)/L,np.real(LB)/L,color=color_shading,alpha=0.5) 
  p1=plt.loglog(k,np.real(PHI)/L, color=color_mean)
  #p2=plt.loglog(k[20:int(np.shape(k1b7)[0]-10500)]*L7/(2.*np.pi*Ncycles7)*5.,3e-10*np.power(k1b7[20:int(np.shape(k1b7)  [0]-10500)]*L7/200.,-10./3.), '--k')
  plt.ylabel(figure_ylabel, fontsize=16) 
  #plt.yticks([1e-20,1e-17,1e-14,1e-11,1e-8,1e-5,1e-2]) #np.arange(1e-20, 1e-2, 1e-4))
  plt.yticks([1e-20,1e-14,1e-8,1e-2]) #np.arange(1e-20, 1e-2, 1e-4))
  plt.grid()
  plt.axis(axes) # axes = [2e-2,4e2,1e-23,2e-4]
  if ytick_label_flag == 0:
    plt.tick_params(
      axis='y',          # changes apply to the x-axis
      #which='both',      # both major and minor ticks are affected
      #bottom='on',      # ticks along the bottom edge are off
      #top='on',         # ticks along the top edge are off
      #labelbottom='off') # labels along the bottom edge are off
      labelleft='off') # labels along the bottom edge are off
  if xtick_label_flag == 0:
    plt.tick_params(
      axis='x',          # changes apply to the x-axis
      #which='both',      # both major and minor ticks are affected
      #bottom='on',      # ticks along the bottom edge are off
      #top='on',         # ticks along the top edge are off
      labelbottom='off') # labels along the bottom edge are off
  #else:
  plt.xlabel(r"$\omega/\Omega$", fontsize=16)
  bg = np.array([1,1,1])  # background of the legend is white
  #colors = ['green'] #,'blue'] #,'green','green']
  # with alpha = .5, the faded color is the average of the background and color
  colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
  plt.legend([0], [legend_name],handler_map={0: LegendObject(colors[0], colors_faded[0])},loc=3)
               # 1: LegendObject(colors[1], colors_faded[1])},loc=1)
               #2: LegendObject(colors[2], colors_faded[2], dashed=True),    
  return

def time_series_plot( t, T, u, color_mean, legend_label, figure_path, figure_name, figure_ylabel, plot_title ):
  fig = plt.figure()
  plotname = figure_path + figure_name 
  plot_title = "z-integrated y-mean du/dz"
  fig = plt.figure() 
  plt.plot(t/T,u,color_mean,label=legend_label); 
  plt.xlabel("t/T"); plt.legend(loc=4); plt.ylabel(figure_ylabel); 
  plt.title(plot_title);
  plt.axis('tight')  
  plt.savefig(plotname,format="png"); 
  plt.close(fig)
  return

def time_series_subplot( t, T, u, color_mean, legend_label , figure_ylabel, tick_label_flag ):
  plt.plot(t/T,u,color_mean,label=legend_label); 
  if tick_label_flag == 0:
    plt.tick_params(
      axis='y',          # changes apply to the x-axis
      #which='both',      # both major and minor ticks are affected
      #bottom='on',      # ticks along the bottom edge are off
      #top='on',         # ticks along the top edge are off
      #labelbottom='off') # labels along the bottom edge are off
      labelleft='off') # labels along the bottom edge are off
  plt.xlabel(r"$\mathrm{t}/\mathrm{T}$", fontsize=16); 
  plt.legend(loc=4); plt.ylabel(figure_ylabel, fontsize=16); 
  plt.axis('tight')  
  return

def plug_solutions(N,T,Uw,nu,Pr,C,time,Nz,H):

 # from inputs:
 kap = nu/Pr # m^2/s, thermometric diffusivity
 omg = 2.0*np.pi/T # rads/s
 #print(N,omg,T)
 thtcrit = ma.asin(omg/N) # radians
 Lex = Uw/omg # m, excursion length
 tht = C*thtcrit # rads
 # Chebyshev grid:
 #kz = np.linspace(1., Nz, num=Nz)
 #z = -np.cos((kz*2.-1.)/(2.*Nz)*np.pi)*H/2.+H/2. # m
 # uniform grid:
 z= np.linspace(0.0 , H, num=Nz) # m 
 dz = z[1]-z[0] # m
 
 d0=((4.*nu**2.)/((N**2.)*(np.sin(tht))**2.))**(1./4.) # Phillips-Wunsch BL thickness
 Bw = Uw*(N**2.0)*np.sin(tht)/omg # forcing amplitude
 Re = Uw**2./(omg*nu) # the true Stokes' Reynolds number (the square of length scale ratio)

 if C < 1.: # subcritcal 
  d1 = np.power( omg*(1.+Pr)/(4.*nu) + \
      np.power((omg*(1.+Pr)/(4.*nu))**2. + \
      Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
  d2 = np.power( omg*(1.+Pr)/(4.*nu) - \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
  L = ((d1-d2)*(2.*nu/Pr+omg*d1*d2))/(omg*d1*d2) # wall-normal buoyancy gradient lengthscale
  u1 = d2*(omg*d1**2.-2.*nu/Pr)/(L*omg*d1*d2) # unitless
  u2 = d1*(2.*nu/Pr-omg*d2**2.)/(L*omg*d1*d2) # unitless
  b1 = d1/L # unitless
  b2 = d2/L # unitless
  alpha1 = (omg*d1**2.-2.*nu/Pr)/(L*omg*d1)
  alpha2 = (2.*nu/Pr-omg*d2**2.)/(L*omg*d2)
  coeffs = (kap,omg,tht,thtcrit,d0,Bw,Re,d1,d2,L,u1,u2,b1,b2,alpha1,alpha2)

 if C > 1.: # supercritical 
  d1 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) + \
       omg*(1.+Pr)/(4.*nu), -1./2.)
  d2 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) - \
       omg*(1.+Pr)/(4.*nu), -1./2.)
  L = np.power(((d1**2.+d2**2.)*(4.*(nu/Pr)**2. + \
      omg**2.*d1**2.*d2**2.))/(omg**2.*d1*d2), 1./4.) 
  u1 = 2.*kap/(d2*omg**2.*L**4.)+d2/(omg*L**4.) # s/m^3
  u2 = d1/(omg*L**4.)-2.*kap/(d1*omg**2.*L**4.) # s/m^3
  b1 = d1/(omg*L**4.) # s/m^3
  b2 = d2/(omg*L**4.) # s/m^3
  alpha1 = u1*(2.*kap*d1 + omg*d2**2.*d1)
  alpha2 = u1*(2.*kap*d2 - omg*d1**2.*d2)
  alpha3 = u2*( omg*d1**2.*d2 - 2.*kap*d2)
  alpha4 = u2*(2.*kap*d1 + omg*d2**2.*d1)
  beta1 = b1*(omg*d1**2.*d2-2.*kap*d2)
  beta2 = b1*(2.*kap*d1+omg*d1*d2**2.)
  beta3 = b2*(2.*kap*d1+omg*d1*d2**2.)
  beta4 = b2*(2.*kap*d2-omg*d1**2.*d2)
  coeffs = (kap,omg,tht,thtcrit,d0,Bw,Re,d1,d2,L,u1,u2,b1,b2,alpha1,alpha2,alpha3,alpha4,beta1,beta2,beta3,beta4)


 u = np.zeros([Nz,1]); b = np.zeros([Nz,1])
 uz = np.zeros([Nz,1]); bz = np.zeros([Nz,1])

 for j in range(0,Nz): 
   u[j,0] = Uw
   uz[j,0] = 0.
   b[j,0] = 0.
   bz[j,0] = 0.

 return coeffs, z, dz, u, uz, b, bz


def inst_solutions(N,T,Uw,nu,Pr,C,time,Nz,H): 

 # from inputs:
 kap = nu/Pr # m^2/s, thermometric diffusivity
 omg = 2.0*np.pi/T # rads/s
 #print(N,omg,T)
 thtcrit = ma.asin(omg/N) # radians
 Lex = Uw/omg # m, excursion length
 tht = C*thtcrit # rads
 # Chebyshev grid:
 #kz = np.linspace(1., Nz, num=Nz)
 #z = -np.cos((kz*2.-1.)/(2.*Nz)*np.pi)*H/2.+H/2. # m
 # uniform grid:
 z= np.linspace(0.0 , H, num=Nz) # m 
 dz = z[1]-z[0] # m
 
 d0=((4.*nu**2.)/((N**2.)*(np.sin(tht))**2.))**(1./4.) # Phillips-Wunsch BL thickness
 Bw = Uw*(N**2.0)*np.sin(tht)/omg # forcing amplitude
 Re = Uw**2./(omg*nu) # the true Stokes' Reynolds number (the square of length scale ratio)

 if C < 1.: # subcritcal 
  d1 = np.power( omg*(1.+Pr)/(4.*nu) + \
      np.power((omg*(1.+Pr)/(4.*nu))**2. + \
      Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
  d2 = np.power( omg*(1.+Pr)/(4.*nu) - \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
  L = ((d1-d2)*(2.*nu/Pr+omg*d1*d2))/(omg*d1*d2) # wall-normal buoyancy gradient lengthscale
  u1 = d2*(omg*d1**2.-2.*nu/Pr)/(L*omg*d1*d2) # unitless
  u2 = d1*(2.*nu/Pr-omg*d2**2.)/(L*omg*d1*d2) # unitless
  b1 = d1/L # unitless
  b2 = d2/L # unitless
  alpha1 = (omg*d1**2.-2.*nu/Pr)/(L*omg*d1)
  alpha2 = (2.*nu/Pr-omg*d2**2.)/(L*omg*d2)
  coeffs = (kap,omg,tht,thtcrit,d0,Bw,Re,d1,d2,L,u1,u2,b1,b2,alpha1,alpha2)

 if C > 1.: # supercritical 
  d1 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) + \
       omg*(1.+Pr)/(4.*nu), -1./2.)
  d2 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) - \
       omg*(1.+Pr)/(4.*nu), -1./2.)
  L = np.power(((d1**2.+d2**2.)*(4.*(nu/Pr)**2. + \
      omg**2.*d1**2.*d2**2.))/(omg**2.*d1*d2), 1./4.) 
  u1 = 2.*kap/(d2*omg**2.*L**4.)+d2/(omg*L**4.) # s/m^3
  u2 = d1/(omg*L**4.)-2.*kap/(d1*omg**2.*L**4.) # s/m^3
  b1 = d1/(omg*L**4.) # s/m^3
  b2 = d2/(omg*L**4.) # s/m^3
  alpha1 = u1*(2.*kap*d1 + omg*d2**2.*d1)
  alpha2 = u1*(2.*kap*d2 - omg*d1**2.*d2)
  alpha3 = u2*( omg*d1**2.*d2 - 2.*kap*d2)
  alpha4 = u2*(2.*kap*d1 + omg*d2**2.*d1)
  beta1 = b1*(omg*d1**2.*d2-2.*kap*d2)
  beta2 = b1*(2.*kap*d1+omg*d1*d2**2.)
  beta3 = b2*(2.*kap*d1+omg*d1*d2**2.)
  beta4 = b2*(2.*kap*d2-omg*d1**2.*d2)
  coeffs = (kap,omg,tht,thtcrit,d0,Bw,Re,d1,d2,L,u1,u2,b1,b2,alpha1,alpha2,alpha3,alpha4,beta1,beta2,beta3,beta4)


 u = np.zeros([Nz,1]); b = np.zeros([Nz,1])
 uz = np.zeros([Nz,1]); bz = np.zeros([Nz,1])

 for j in range(0,Nz): 

  if C < 1.: # subcritical slopes
   u[j,0] = Uw*np.real( (u1*np.exp(-(1.+1j)*z[j]/d1) + \
               u2*np.exp(-(1.+1j)*z[j]/d2) - 1.)*np.exp(1j*omg*time) )
   uz[j,0] = - Uw*np.real( (u1*(1.+1j)/d1*np.exp(-(1.+1j)*z[j]/d1) + \
               u2*(1.+1j)/d2*np.exp(-(1.+1j)*z[j]/d2) )*np.exp(1j*omg*time) )
   b[j,0] = Bw*np.real( (b1*np.exp(-(1.0+1j)*z[j]/d1) - \
               b2*np.exp(-(1.+1j)*z[j]/d2) - 1.)*1j*np.exp(1j*omg*time) )
   bz[j,0] = Bw*np.real( ( -(1.0+1j)/d1*b1*np.exp(-(1.0+1j)*z[j]/d1) + \
             (1.+1j)/d2*b2*np.exp(-(1.+1j)*z[j]/d2) )*1j*np.exp(1j*omg*time) )

  if C > 1.: # supercritical slopes
   u[j,0] = Uw*np.real( ( ( alpha1 + 1j*alpha2 )*np.exp((1j-1.0)*z[j]/d2)+ \
               ( alpha3 + 1j*alpha4 )*np.exp(-(1j+1.)*z[j]/d1) - 1. )*np.exp(1j*omg*time) )

   uz[j,0] = Uw*np.real( ( (1j-1.0)/d2*( alpha1 + 1j*alpha2 )*np.exp((1j-1.0)*z[j]/d2) \
               -(1j+1.)/d1*(alpha3 + 1j*alpha4 )*np.exp(-(1j+1.)*z[j]/d1) )*np.exp(1j*omg*time) )

   b[j,0] = Bw*np.real( ( ( beta1 + 1j*beta2 )*np.exp(-(1j+1.0)*z[j]/d1)+ \
               ( beta1 + 1j*beta2 )*np.exp((1j-1.0)*z[j]/d2) -1. )*1j*np.exp(1j*omg*time) )

   bz[j,0] = Bw*np.real( ( -(1j+1.0)/d1*( beta1 + 1j*beta2 )*np.exp(-(1j+1.0)*z[j]/d1)+ \
               (1j-1.0)/d2*( beta1 + 1j*beta2 )*np.exp((1j-1.0)*z[j]/d2) )*1j*np.exp(1j*omg*time) )

 return coeffs, z, dz, u, uz, b, bz


def make_Lap_inv(dz,Nz,K2):
 # 2nd order accurate truncation
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - K2 - 2./(dz**2.)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(dz**2.)
 La = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 La[0,0:4] = [ -K2 + 2./(dz**2.), -5./(dz**2.), 4./(dz**2.), -1./(dz**2.) ] # lower (wall) BC
 La[Nz-1,Nz-4:Nz] = [ -1./(dz**2.), 4./(dz**2.), -5./(dz**2.), -K2 + 2./(dz**2.) ] # upper (far field) BC
 La_inv = np.linalg.inv(La)
 """ 
 print(La[1,0:10])
 #print(La[38,34:40])
 #print(1./(dz**2.))
 #print(-K2)
 print(-K2-2./(dz**2.))
 print(1./(dz**2.))
 print(-K2+2./(dz**2.))
 print(-5./(dz**2.))
 print(4./(dz**2.))
 print(-1./(dz**2.))
 """
 return La_inv

def make_e(dz,Nz,C,tht,k):
 # 2nd order accurate truncation
 cottht = np.cos(tht)/np.sin(tht)
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - 1j*k*C**2.
 for j in range(0,Nz-1):
  diagNzm1[j] = cottht*C**2./(2.*dz)
 #print( np.diag(diagNzm1,1) )
 #print(- np.diag(diagNzm1,-1) )
 e = np.diag(diagNzm1,1) + np.diag(diagNz,0) - np.diag(diagNzm1,-1)
 # now add upper and lower BCs:
 #r[0,0:3] = [ -C**2.*( 1j*k + 3.*cottht/(2.*dz) ), 2.*C**2.*cottht/dz, -C**2.*cottht/(2.*dz) ] # lower (wall) BC
 #r[Nz-1,Nz-3:Nz] = [ C**2.*cottht/(2.*dz), -2.*C**2.*cottht/dz, C**2.*( -1j*k + 3.*cottht/(2.*dz) ) ] # upper (far field) BC
 #print(- 1j*k*C**2.)
 #print(cottht*C**2./(2.*dz))
 #print(e)
 return e



def make_partial_z(dz,Nz):
 # 2nd order accurate truncation
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(2.*dz)
 pz = np.diag(diagNzm1,k=1) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 pz[0,0:3] = [ -3./(2.*dz), 2./dz, -1./(2.*dz) ] # lower (wall) BC
 pz[Nz-1,Nz-3:Nz] = [ 1./(2.*dz), -2./dz, 3./(2.*dz) ] # upper (far field) BC
 return pz


def make_r(dz,Nz,C,tht,k):
 # 2nd order accurate truncation
 cottht = np.cos(tht)/np.sin(tht)
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - 1j*k*C**2.
 for j in range(0,Nz-1):
  diagNzm1[j] = cottht*C**2./(2.*dz)
 r = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 r[0,0:3] = [ -C**2.*( 1j*k + 3.*cottht/(2.*dz) ), 2.*C**2.*cottht/dz, -C**2.*cottht/(2.*dz) ] # lower (wall) BC
 r[Nz-1,Nz-3:Nz] = [ C**2.*cottht/(2.*dz), -2.*C**2.*cottht/dz, C**2.*( -1j*k + 3.*cottht/(2.*dz) ) ] # upper (far field) BC
 return r



def make_stationary_matrices(dz,Nz,C,K2,tht,k):
 La_inv = make_Lap_inv(dz,Nz,K2)
 pz = make_partial_z(dz,Nz)
 r = make_r(dz,Nz,C,tht,k)
 P4 = np.dot(La_inv,r)
 return La_inv, pz, P4

def check_matrix(self,string):
 if np.any(np.isnan(self)):
  print('NaN detected in '+ string)
 if np.any(np.isinf(self)):
  print('Inf detected in '+ string)
 return

def make_A43(Nz,Bz,tht):
 cottht = np.cos(tht)/np.sin(tht)
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = Bz[j]
 #print(Bz[0],Bz[1],Bz[Nz-1]-1e-3**2.*np.cos(tht))
 A43 = - np.diag(diagNz,k=0) - np.eye(Nz,Nz,0,dtype=complex)*cottht
 return A43

def make_A13(Nz,Uz,k,P3):
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(1,Nz-1):
  diagNz[j] = -Uz[j]
 A13 = np.diag(diagNz,k=0) + 1j*k*P3 
 #A13[0,0] = 0.+0.*1j
 #A13[Nz-1,Nz-1] = 0.+0.*1j # BCs on top/bottom w applied
 #print(A13[0,0],A13[Nz-1,Nz-1])
 #print(A13[1,1],A13[Nz-2,Nz-2])
 #print(Uz[0],Uz[Nz-1])
 # near bottom values are large!
 return A13

def make_A34(Nz,C,tht,dzP4):
 cottht = np.cos(tht)/np.sin(tht)
 A34 = np.eye(Nz,Nz,0,dtype=complex)*C**2.*cottht - dzP4 # check eye <--------------||
 # check the buoyancy boundary condition!
 #print(A34[0,0],A34[1,1])
 return A34

def make_A14(Nz,k,P4,C):
 #G = np.eye(Nz,Nz,0,dtype=complex)*C**2.
 A14 = 1j*k*P4 + np.eye(Nz,Nz,0,dtype=complex)*C**2.
 #print(C**2.,G[0,0],G[Nz-1,Nz-1])
 return A14

def make_D4(dz,Nz,U,k,l,Re,Pr):
 # 2nd order accurate truncation
 K2 = k**2.+l**2.
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - K2 - 2./(dz**2.)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(dz**2.)
 D =  np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1) 
 # now add upper and lower BCs:
 D[0,0:4] = [ -K2 + 2./(dz**2.), -5./(dz**2.), 4./(dz**2.), -1./(dz**2.) ]  # lower (wall) BC
 D[Nz-1,Nz-4:Nz] = [ -1./(dz**2.), 4./(dz**2.), -5./(dz**2.), -K2 + 2./(dz**2.) ]  # upper (far field) BC
 D4 = np.eye(Nz,Nz,0,dtype=complex)*1j*k*U + D/(Re*Pr)
 return D4

def make_DI(dz,Nz,U,k,l,Re):
 # 2nd order accurate truncation
 K2 = k**2.+l**2.
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = - K2 - 2./(dz**2.)
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(dz**2.)
 D =  np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1) 
 # now add upper and lower BCs:
 D[0,0:4] = [ -K2 + 2./(dz**2.), -5./(dz**2.), 4./(dz**2.), -1./(dz**2.) ]  # lower (wall) BC
 D[Nz-1,Nz-4:Nz] = [ -1./(dz**2.), 4./(dz**2.), -5./(dz**2.), -K2 + 2./(dz**2.) ]  # upper (far field) BC
 DI = np.eye(Nz,Nz,0,dtype=complex)*1j*k*U + D/Re
 return DI

def make_DI_old(dz,Nz,U,k,Re):
 # U needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k*U[j] - 1./Re*(K2 + 2./(dz**2.)) # ,nt[itime]
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(Re*dz**2.)
 DI = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 DI[0,0:4] = [1j*k*U[j] - 1./Re*(K2 - 2./(dz**2.)), -5./(Re*dz**2.), 4./(Re*dz**2.), -1./(Re*dz**2.) ] # lower (wall) BC
 DI[Nz-1,Nz-4:Nz] = [-1./(Re*dz**2.), 4./(Re*dz**2.), -5./(Re*dz**2.), 1j*k*U[j] - 1./Re*(K2 - 2./(dz**2.)) ] # upper (far field) BC
 return DI

def make_D4_old(dz,Nz,U,k,Re,Pr):
 # U needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 diagNzm1 = np.zeros([Nz-1], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k*U[j] - 1./(Re*Pr)*(K2 - 2./(dz**2.))
 for j in range(0,Nz-1):
  diagNzm1[j] = 1./(Re*Pr*dz**2.)
 D4 = np.diag(diagNzm1,k=1) + np.diag(diagNz,k=0) + np.diag(diagNzm1,k=-1)
 # now add upper and lower BCs:
 D4[0,0:4] = [1j*k*U[j] - 1./(Re*Pr)*(K2 - 2./(dz**2.)), -5./((Re*Pr)*dz**2.), 4./((Re*Pr)*dz**2.), -1./((Re*Pr)*dz**2.) ] # lower (wall) BC
 D4[Nz-1,Nz-4:Nz] = [-1./((Re*Pr)*dz**2.), 4./((Re*Pr)*dz**2.), -5./((Re*Pr)*dz**2.), 1j*k*U[j] - 1./(Re*Pr)*(K2 - 2./(dz**2.)) ] # upper (far field) BC
 return D4

def make_d(k,Uz,Nz):
 # Uz needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(1,Nz-1):
  diagNz[j] = 1j*k*Uz[j]
 d = np.diag(diagNz,k=0) # BCs on top/bottom w applied
 return d


def make_q(k,Uz):
 # Uz needs to be a vector length Nz, input U[:,nt[itime]]
 diagNz = np.zeros([Nz], dtype=complex)
 for j in range(0,Nz):
  diagNz[j] = 1j*k*Uz[j]
 q = np.diag(diagNz,k=0) # no BCs needed
 return q

def make_transient_matrices(dz,Nz,U,k,Re,Pr,Uz,La_inv):
 DI = make_DI(dz,Nz,U,k,Re)
 D4 = make_D4(dz,Nz,U,k,Re,Pr)
 q = make_q(k,Uz)
 P3 = np.dot(La_inv,q)
 if np.any(np.isnan(DI)):
  print('NaN detected in DI')
 if np.any(np.isinf(DI)):
  print('Inf detected in DI')
 if np.any(np.isnan(D4)):
  print('NaN detected in D4')
 if np.any(np.isinf(D4)):
  print('Inf detected in D4')
 if np.any(np.isnan(q)):
  print('NaN detected in q')
 if np.any(np.isinf(q)):
  print('Inf detected in q')
 if np.any(np.isnan(P3)):
  print('NaN detected in P3')
 if np.any(np.isinf(P3)):
  print('Inf detected in P3')
 return DI, D4, P3


def populate_A(DI, D4, P3, P4, pz, dz, Bz, Uz, cottht, k, l):
 A = np.zeros([int(4*Nz),int(4*Nz)], dtype=complex)
 A11 = DI
 A12 = np.zeros([Nz,Nz])
 A13 = -np.diag(Uz[:],k=0) + 1j*k*P3
 A14 = np.diag(C**2.*np.ones([Nz]),k=0)  + 1j*k*P4
 A21 = np.zeros([Nz,Nz])
 A22 = DI 
 A23 = 1j*l*P3
 A24 = 1j*l*P4
 A31 = np.zeros([Nz,Nz])
 A32 = np.zeros([Nz,Nz])
 A33 = DI - np.dot(pz,P3)
 A34 = np.diag(C**2.*cottht*np.ones([Nz]),k=0) - np.dot(pz,P4)
 A41 = np.identity(Nz)
 A42 = np.zeros([Nz,Nz])
 A43 = -np.diag(Bz[:] + cottht*np.ones([Nz]),k=0) 
 A44 = D4
 A1 = np.hstack((A11,A12,A13,A14))
 A2 = np.hstack((A21,A22,A23,A24))
 A3 = np.hstack((A31,A32,A33,A34))
 A4 = np.hstack((A41,A42,A43,A44))
 A = np.vstack((A1,A2,A3,A4))
 if np.any(np.isnan(A)):
  print('NaN detected in A in populate_A()')
 if np.any(np.isinf(A)):
  print('Inf detected in A in populate_A()')
 return A


def inst_construct_A(La_inv,pz,P4,dz,Nz,U,Uz,Bz,k,l,Re,Pr):

 (DI, D4, P3) = make_transient_matrices(dz,Nz,U[:,0],k,Re,Pr,Uz[:,0],La_inv)

 # construct A:
 A = populate_A(DI, D4, P3, P4, pz, dz, Bz[:,0], Uz[:,0], cottht, k, l)

 # tests:
 #print(sum(sum(A[0:Nz,0:Nz]-DI)))
 #print(sum(sum(A[(2*Nz):(3*Nz),(2*Nz):(3*Nz)]-( DI - np.dot(pz,P3)) )))
 #print(sum(sum(A[0:Nz,int(2*Nz):int(3*Nz)]+np.diag(Uz[:,0],k=0) - 1j*k*P3)))
 
 return A

"""
def rk4_test(n, Nz, N, omg, tht, nu, kap, U, t, z, dz, l, k, Phi ):

 A = [[-1.,np.cos(omg*t[n])],[0.,-2.]]

 check_matrix(Am,'Am')

 # Runge-Kutta coefficients
 krk = np.dot(Am,Phi) 
 
 check_matrix(krk,'krk') 

 return
"""

def rk4_old(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,time,Phi): 
 # constructs A and take the dot product A*Phi

 # instantaneous mean flow solutions at time t
 (coeffs, z, dz, U, Uz, B, Bz) = inst_solutions(N,T,Uw,nu,Pr,C,time,Nz,H)
 #(coeffs, z, dz, U, Uz, B, Bz) = plug_solutions(N,T,Uw,nu,Pr,C,time,Nz,H) # <-------------------------|||
 
 # construction of "A" matrix at time t
 A = inst_construct_A(La_inv,pz,P4,dz,Nz,U,Uz,Bz,k,l,coeffs[6],Pr)
 if np.any(np.isnan(A)):
  print('NaN detected in A in inst_construct_A()')
 if np.any(np.isinf(A)):
  print('Inf detected in A in inst_construct_A()')
 if np.any(np.isnan(Phin)):
  print('NaN detected in Phin in inst_construct_A()')
 if np.any(np.isinf(Phin)):
  print('Inf detected in Phin in inst_construct_A()')

 # Runge-Kutta coefficients
 krk = np.dot(A,Phi) 
 if np.any(np.isnan(krk)):
  print('NaN detected in Runge-Kutta coefficient in rk4()')
 if np.any(np.isinf(krk)):
  print('Inf detected in Runge-Kutta coefficient in rk4()')
 
 return krk

def rk4(n, Nz, N, omg, tht, nu, kap, U, t, z, dz, l, k, Phi ):
 
 L = U/omg
 Re = omg*L**2./nu
 Pr = nu/kap
 #print(t[n])

 # Phillips (1970) / Wunsch (1970) steady non-rotating solution
 #( b0, u0, bz0, uz0 ) = steady_nonrotating_solution( N, omg, tht, nu, kap, [t[n]], z )
 # non-rotating oscillating solution, Baidulov (2010):
 ( b, u, bz, uz ) = xforcing_nonrotating_solution( U, N, omg, tht, nu, kap, [t[n]], z )

 # pressure matrices:
 #d = make_d(k,uz0 + uz,Nz) 
 d = make_d(k,uz,Nz) # add the mean back in?
 e = make_e(dz,Nz,N*np.sin(tht)/omg,tht,k)
 La_inv = make_Lap_inv(dz,Nz,l**2.+k**2.)
 partial_z = make_partial_z(dz,Nz)
 P3 = np.dot(La_inv,d)
 P4 = np.dot(La_inv,e)
 #print(P4[0,0],P4[1,1])
 #print(np.shape(P3),np.shape(P4))
 dzP3 = np.dot(partial_z,P3)
 dzP4 = np.dot(partial_z,P4)

 check_matrix(e,'e')
 check_matrix(d,'d')
 check_matrix(La_inv,'La_inv')
 check_matrix(partial_z,'partial_z')
 check_matrix(P3,'P3')
 check_matrix(P4,'P4')
 check_matrix(dzP3,'dzP3')
 check_matrix(dzP4,'dzP4')
 
 # advective-diffusive matrices:
 DI = make_DI(dz,Nz,U,k,l,Re)
 D4 = make_D4(dz,Nz,U,k,l,Re,Pr)
  
 A11 = DI
 A12 = np.zeros([Nz,Nz],dtype=complex) 
 #A13 = make_A13(Nz,uz0 + uz,k,P3)
 A13 = make_A13(Nz,uz,k,P3) # add the mean back in?
 A14 = make_A14(Nz,k,P4,N*np.sin(tht)/omg)
 A21 = np.zeros([Nz,Nz],dtype=complex)
 A22 = DI
 A23 = 1j*l*P3
 A24 = 1j*l*P4
 A31 = np.zeros([Nz,Nz],dtype=complex)
 A32 = np.zeros([Nz,Nz],dtype=complex)
 A33 = DI - dzP3
 A34 = make_A34(Nz,N*np.sin(tht)/omg,tht,dzP4)
 A41 = np.eye(Nz,Nz,0,dtype=complex)
 A42 = np.zeros([Nz,Nz],dtype=complex)
 #A43 = make_A43(Nz,bz0 + bz + N**2.*np.cos(tht),tht) 
 A43 = make_A43(Nz,bz,tht) # add the mean back in?
 A44 = D4

 check_matrix(DI,'DI')
 check_matrix(D4,'D4')
 check_matrix(A13,'A13')
 check_matrix(A14,'A14')
 check_matrix(A33,'A33')
 check_matrix(A34,'A34')
 check_matrix(A43,'A43')

 A1 = np.concatenate((A11,A12,A13,A14),axis=1)
 A2 = np.concatenate((A21,A22,A23,A24),axis=1)
 A3 = np.concatenate((A31,A32,A33,A34),axis=1)
 A4 = np.concatenate((A41,A42,A43,A44),axis=1)
 Am = np.concatenate((A1,A2,A3,A4),axis=0)
 
 check_matrix(Am,'Am')

 # Runge-Kutta coefficients
 krk = np.dot(Am,Phi) 
 
 check_matrix(krk,'krk')

 return krk



def rk4_test( alpha, beta, omg, t, P ):
 #print(-alpha)
 #print(-beta)
 #print(t[n])
 #print(alpha,beta)
 A0 = [[-alpha,-np.exp(1j*omg*t)],[0.,-beta]]
 #print(Am)
 #check_matrix(Am,'Am')

 # Runge-Kutta coefficients
 krk = np.dot(A0,P) 
 
 #check_matrix(krk,'krk')

 return krk

def ordered_prod( alpha, beta, omg, t , dt):
 P0 = np.exp([[-alpha,-np.exp(1j*omg*t)],[0.,-beta]]*np.ones([2,2])*dt)
 return P0

def steady_nonrotating_solution( N, omg, tht, nu, kap, t, z ):
 # Phillips (1970) / Wunsch (1970) non-rotating steady solution

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap

 d0=((4.*nu**2.)/((N**2.)*(np.sin(tht))**2.))**(1./4.) 

 b0 = np.zeros([Nz,Nt])
 u0 = np.zeros([Nz,Nt])
 bz0 = np.zeros([Nz,Nt])
 uz0 = np.zeros([Nz,Nt])

 for i in range(0,Nz):
  for j in range(0,Nt):
   Z = z[i]/d0
   u0[i,j] = 2.0*nu/(Pr*d0)*(np.tan(tht)**(-1.))*np.exp(-Z)*np.sin(Z)
   b0[i,j] = (N**2.)*d0*np.cos(tht)*np.exp(-Z)*np.cos(Z)
   bz0[i,j] = -(N**2.)*d0*np.cos(tht)*np.exp(-Z)*( np.sin(Z) + np.cos(Z) )/d0
   uz0[i,j] = 2.0*nu/(Pr*d0**2.)*(np.tan(tht)**(-1.))*np.exp(-Z)*(np.cos(Z)-np.sin(Z))

 return b0, u0, bz0, uz0

def xforcing_nonrotating_solution( U, N, omg, tht, nu, kap, t, z ):

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap
 
 thtcrit = ma.asin(omg/N) # radians 
 criticality = tht/thtcrit
 #print(criticality)

 Bs = U*(N**2.0)*np.sin(tht)/omg
 d1 = 0.; d2 = 0.; u1 = 0.; u2 = 0.; b1 = 0.; b2 = 0.; Ld = 0.

 if criticality > 1.0:
   d1 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) + \
       omg*(1.+Pr)/(4.*nu), -1./2.)
   d2 = np.power( np.power( (omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2.) - \
       omg*(1.+Pr)/(4.*nu), -1./2.)
   Ld = np.power(((d1**2.+d2**2.)*(4.*(nu/Pr)**2. + \
      omg**2.*d1**2.*d2**2.))/(omg**2.*d1*d2), 1./4.) 
   u1 = 2.*kap/(d2*omg**2.*Ld**4.)+d2/(omg*Ld**4.) # s/m^3
   u2 = d1/(omg*Ld**4.)-2.*kap/(d1*omg**2.*Ld**4.) # s/m^3
   b1 = d1/(omg*Ld**4.) # s/m^3
   b2 = d2/(omg*Ld**4.) # s/m^3
   alpha1 = (omg*d1**2.-2.*nu/Pr)/(Ld*omg*d1)
   alpha2 = (2.*nu/Pr-omg*d2**2.)/(Ld*omg*d2)

 elif criticality < 1.0:
   d1 = np.power( omg*(1.+Pr)/(4.*nu) + \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
   d2 = np.power( omg*(1.+Pr)/(4.*nu) - \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(tht)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
   Ld = ((d1-d2)*(2.*nu/Pr+omg*d1*d2))/(omg*d1*d2) # wall-normal buoyancy gradient lengthscale
   u1 = d2*(omg*d1**2.-2.*nu/Pr)/(Ld*omg*d1*d2) # unitless
   u2 = d1*(2.*nu/Pr-omg*d2**2.)/(Ld*omg*d1*d2) # unitless
   b1 = d1/Ld # unitless
   b2 = d2/Ld # unitless
   alpha1 = u1*(2.*kap*d1 + omg*d2**2.*d1)
   alpha2 = u1*(2.*kap*d2 - omg*d1**2.*d2)
   alpha3 = u2*( omg*d1**2.*d2 - 2.*kap*d2)
   alpha4 = u2*(2.*kap*d1 + omg*d2**2.*d1)
   beta1 = b1*(omg*d1**2.*d2-2.*kap*d2)
   beta2 = b1*(2.*kap*d1+omg*d1*d2**2.)
   beta3 = b2*(2.*kap*d1+omg*d1*d2**2.)
   beta4 = b2*(2.*kap*d2-omg*d1**2.*d2)
  
 elif criticality == 1.:
   print('ERROR: critical slope')
   return

 b = np.zeros([Nz,Nt])
 u = np.zeros([Nz,Nt])
 uz = np.zeros([Nz,Nt])
 bz = np.zeros([Nz,Nt])
 
 for i in range(0,Nz):
   for j in range(0,Nt):

     if criticality < 1.:
       u[i,j] = U*np.real( (u1*np.exp(-(1.+1j)*z[i]/d1) + \
                u2*np.exp(-(1.+1j)*z[i]/d2) - 1.)*np.exp(1j*omg*t[j]) )
       uz[i,j] = - U*np.real( (u1*(1.+1j)/d1*np.exp(-(1.+1j)*z[i]/d1) + \
               u2*(1.+1j)/d2*np.exp(-(1.+1j)*z[i]/d2) )*np.exp(1j*omg*t[j]) )
       b[i,j] = Bs*np.real( (b1*np.exp(-(1.0+1j)*z[i]/d1) - \
                b2*np.exp(-(1.+1j)*z[i]/d2) - 1.)*1j*np.exp(1j*omg*t[j]) )
       bz[i,j] = Bs*np.real( ( -(1.0+1j)/d1*b1*np.exp(-(1.0+1j)*z[i]/d1) + \
             (1.+1j)/d2*b2*np.exp(-(1.+1j)*z[i]/d2) )*1j*np.exp(1j*omg*t[j]) )

     if criticality > 1.:
       u[i,j] = U*np.real( (u1*(2.*kap*d1+omg*d1*d2**2.+ \
                1j*(2.*kap*d2-omg*d1**2.*d2))*np.exp((1j-1.0)*z[i]/d2)+ \
                u2*(omg*d1**2.*d2-2.*kap*d2+ \
                1j*(2.0*kap*d1+omg*d1*d2**2.))*np.exp(-(1j+1.)*z[i]/d1)- \
                1.)*np.exp(1j*omg*t[j]) )
       uz[i,j] = U*np.real( ( (1j-1.0)/d2*( alpha1 + 1j*alpha2 )*np.exp((1j-1.0)*z[i]/d2) \
               -(1j+1.)/d1*(alpha3 + 1j*alpha4 )*np.exp(-(1j+1.)*z[i]/d1) )*np.exp(1j*omg*t[j]) )
       b[i,j] = Bs*np.real( (b1*(omg*d1**2.*d2sp-2.*kap*d2+ \
                1j*(2.*kap*d1+omg*d1*d2**2.))*np.exp(-(1j+1.0)*z[j]/d1)+ \
                b2*(2.*kap*d1+omg*d1*d2**2.+ \
                1j*(2.*kap*d2-omg*d1**2.*d2))*np.exp((1j-1.0)*z[i]/d2)- \
                -1.)*1j*np.exp(1j*omg*t[j]) )
       bz[i,j] = Bs*np.real( ( -(1j+1.0)/d1*( beta1 + 1j*beta2 )*np.exp(-(1j+1.0)*z[i]/d1)+ \
               (1j-1.0)/d2*( beta1 + 1j*beta2 )*np.exp((1j-1.0)*z[i]/d2) )*1j*np.exp(1j*omg*t[j]) )

 return  b, u, bz, uz

def steady_rotating_solution( f, N, omg, tht, nu, kap, t, z ):
 # Wunsch (1970) rotating, steady solution

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap
 cot = np.cos(tht) / np.sin(tht)

 d0 = ( ( f*np.cos(tht) / (2.*nu) )**2. + ( N*np.sin(tht)/2. )**2./(nu*kap) )**(-1./4.)
 vg = -Pr*d0*f*cot # geostrophic far field velocity

 b0 = np.zeros([Nz,Nt])
 u0 = np.zeros([Nz,Nt])
 v0 = np.zeros([Nz,Nt])

 for i in range(0,Nz):
  for j in range(0,Nt):
   Z = z[i]/d0
   b0[i,j] = N**2.*d0*np.cos(tht)*np.exp(-Z)*np.cos(Z)
   u0[i,j] = 2.*kap*cot/d0*np.exp(-Z)*np.sin(Z)
   v0[i,j] = Pr*f*d0*cot*np.exp(-Z)*np.cos(Z) + vg

 return b0, u0, v0

def contour_bounds_2d( self ):
 # self is an array.
 # This function finds the absolute maximum and returns it. 
 cb = 0. # contour bound (such that 0 will be the center of the colorbar)
 if abs(np.amin(self)) > abs(np.amax(self)):
    cb = abs(np.amin(self))
 if abs(np.amax(self)) > abs(np.amin(self)):
    cb = abs(np.amax(self))
 return cb

def zplot( z, H, self, name1, name2 ):
 
 fig = plt.figure() #figsize=(8,4))
 plotname = figure_path + name1 #'/u0_nonrotating.png'
 CP=plt.plot(self,z/H,'b'); 
 plt.xlabel(name2); #plt.legend(loc=4); 
 plt.ylabel('z/H')
 plt.savefig(plotname,format="png"); 
 plt.close(fig)
 
 return

def logzplot( z, H, self, name1, name2 ):
 
 fig = plt.figure() #figsize=(8,4))
 plotname = figure_path + name1 #'/u0_nonrotating.png'
 CP=plt.semilogy(self,z/H,'b'); 
 plt.xlabel(name2); #plt.legend(loc=4); 
 plt.ylabel('z/H')
 plt.savefig(plotname,format="png"); 
 plt.close(fig)
 
 return

def contour_plots( T0, Z0, u0, H, Nc, cmap1, name1, name2 ):

 #u0cb = contour_bounds_2d( u0 )
 #ctrs = np.linspace(-u0cb,u0cb,num=Nc)
 ctrs = np.linspace(np.amin(u0),np.amax(u0),num=Nc)

 fig = plt.figure(figsize=(8,4))
 plotname = figure_path + name1 #'/u0.png'
 CP=plt.contourf(T0,Z0,u0,ctrs,cmap=cmap1); 
 plt.xlabel('t/T'); #plt.legend(loc=4); 
 plt.ylabel('z')
 fig.colorbar(CP)
 plt.savefig(plotname,format="png"); 
 plt.close(fig)

 fig = plt.figure(figsize=(8,4))
 plotname = figure_path + name2 #'/u0_zoom.png'
 CP=plt.contourf(T0,Z0,u0,ctrs,cmap=cmap1); 
 plt.xlabel('t/T'); #plt.legend(loc=4); 
 plt.ylabel('z')
 fig.colorbar(CP)
 plt.axis([0.,1.,0.,H/500.])
 plt.savefig(plotname,format="png"); 
 plt.close(fig)

 return

def xforcing_rotating_solution( U, f, N, omg, tht, nu, kap, t, z ):

 Nt = np.shape(t)[0]
 Nz = np.shape(z)[0]
 Pr = nu / kap
 L = U / omg
 sqfcos = ( f * np.cos( tht ) )**2.
 sqNsin = ( N * np.sin( tht ) )**2.

 A = L*(omg**2. - sqfcos - sqNsin )
 
 # characteristic equation coefficients:
 a4 = - ( 2.*omg/nu + omg/kap ) 
 a2 = - (omg/nu)**2. - 2.*omg**2./(nu*kap) + (f*np.cos(tht)/nu)**2. + (N*np.sin(tht))**2./(nu*kap) 
 a0 =  omg / kap * ( (omg/nu)**2. - (f*np.cos(tht)/nu)**2. - (N*np.sin(tht)/nu)**2. ) 
 ap = ( A * N**2. * np.sin(tht) ) / ( omg**2. - sqfcos - sqNsin ) 

 # solution coefficients
 beta = ( -27.*1j*a0 + 9.*1j*a2*a4 + 2.*1j*(a4**3.) + \
           3.*np.sqrt(3.)*np.sqrt( -27.*a0**2. + 4.*a2**3. + \
           18.*a0*a2*a4 + a2**2.*a4**2. + 4.*a0*a4**3. ) )**(1./3.)
 phi1 = beta/(3.*2.**(1./3.)) - (2.)**(1./3.) * (a4**2. + 3.*a2) / (3.*beta) - 1j*a4/3. 
 phi2 = - (1. - 1j*np.sqrt(3.)) * beta / (6.*2.**(1./3.)) +  \
          (1. + 1j*np.sqrt(3.)) * (a4**2. + 3.*a2) / (3.*2**(2./3.)*beta) - 1j*a4/3. 
 phi3 = - (1. + 1j*np.sqrt(3.)) * beta / (6.*2.**(1./3.)) +  \
         (1. - 1j*np.sqrt(3.)) * (a4**2. + 3.*a2) / (3.*2**(2./3.)*beta) - 1j*a4/3. 

 """
 ups = kap**2.*nu*( np.sqrt(phi2*phi3)*phi1 + np.sqrt(phi1*phi3)*phi2 + \
                    np.sqrt(phi1*phi2)*phi3 ) + 1j*kap*nu*omg*( np.sqrt(phi1*phi2) + \
                    np.sqrt(phi1*phi3) + np.sqrt(phi2*phi3) + phi1 + phi2 + phi3 ) + \
                    nu*omg**2. + kap*(N*np.sin(tht))**2.
 
 c2 = - ( 1./( ups*(np.sqrt(phi1)-np.sqrt(phi2))*(np.sqrt(phi1)-np.sqrt(phi3)) ) ) * ( \
        A*kap*N**2.*np.sqrt(phi2*phi3)*np.sin(tht) + \
        1j*A*N**2.*omg*np.sin(tht) + \
        kap*N**2.*ap*np.sqrt(phi2*phi3)*np.sin(tht)**2. + \
        1j*kap*nu*ap*omg*( np.sqrt(phi3)*phi2**(3./2.) + phi3*phi2 + np.sqrt(phi2)*phi3**(3./2.) ) + \
        nu*ap*omg**2.*np.sqrt(phi2*phi3) )


 c4 =   ( 1./( ups*(np.sqrt(phi1)-np.sqrt(phi2))*(np.sqrt(phi2)-np.sqrt(phi3)) ) ) * ( \
        A*kap*N**2.*np.sqrt(phi1*phi3)*np.sin(tht) + \
        1j*A*N**2.*omg*np.sin(tht) + \
        kap*N**2.*ap*np.sqrt(phi1*phi3)*np.sin(tht)**2. + \
        1j*kap*nu*ap*omg*( np.sqrt(phi3)*phi1**(3./2.) + phi3*phi1 + np.sqrt(phi1)*phi3**(3./2.) ) + \
        nu*ap*omg**2.*np.sqrt(phi1*phi3) )


 c6 = - ( 1./( ups*(np.sqrt(phi1)-np.sqrt(phi3))*(np.sqrt(phi2)-np.sqrt(phi3)) ) ) * ( \
        A*kap*N**2.*np.sqrt(phi1*phi2)*np.sin(tht) + \
        1j*A*N**2.*omg*np.sin(tht) + \
        kap*N**2.*ap*np.sqrt(phi1*phi2)*np.sin(tht)**2. + \
        1j*kap*nu*ap*omg*( np.sqrt(phi2)*phi1**(3./2.) + phi2*phi1 + np.sqrt(phi1)*phi2**(3./2.) ) + \
        nu*ap*omg**2.*np.sqrt(phi1*phi2) )
 """

 E = np.zeros([3,3],dtype= complex)
 E[0,0] = omg + 1j*kap*phi1
 E[0,1] = omg + 1j*kap*phi2
 E[0,2] = omg + 1j*kap*phi3
 E[1,0] = 1j*omg*phi1*(kap+nu) - kap*nu*phi1**2. - (N*np.sin(tht))**2. + omg**2.
 E[1,1] = 1j*omg*phi2*(kap+nu) - kap*nu*phi2**2. - (N*np.sin(tht))**2. + omg**2.
 E[1,2] = 1j*omg*phi3*(kap+nu) - kap*nu*phi3**2. - (N*np.sin(tht))**2. + omg**2.
 E[2,0] = -np.sqrt(phi1)
 E[2,1] = -np.sqrt(phi2)
 E[2,2] = -np.sqrt(phi3)

 Y = np.zeros([3,1],dtype= complex)
 Y[0] = -omg*ap
 Y[1] = A*N**2.*np.sin(tht) - ap*( omg**2. - (N*np.sin(tht))**2. )
 Y[2] = 0.
 C = np.dot(np.linalg.inv(E),Y)
 c2 = C[0] # solution coefficient
 c4 = C[1] # solution coefficient
 c6 = C[2] # solution coefficient

 u1 = c2*(omg + 1j*kap*phi1)
 u2 = c4*(omg + 1j*kap*phi2)
 u3 = c6*(omg + 1j*kap*phi3)

 v1 = c2*(1j*omg*phi1*(kap+nu)-kap*nu*phi1**2.-(N*np.sin(tht))**2.+omg**2.)
 v2 = c4*(1j*omg*phi2*(kap+nu)-kap*nu*phi2**2.-(N*np.sin(tht))**2.+omg**2.)
 v3 = c6*(1j*omg*phi3*(kap+nu)-kap*nu*phi3**2.-(N*np.sin(tht))**2.+omg**2.) 

 # boundary condition checks:
 if abs(np.real(u1+u2+u3+ap*omg)) >= 1e-16:
  print('Wall boundary condition failure for u(z,t)')
  return
 if abs(np.real(v1+v2+v3-A*N**2.*np.sin(tht) + ap*(omg**2.-(N*np.sin(tht))**2) ) ) >= 1e-16:
  print('Wall boundary condition failure for v(z,t)')
  return 
 if abs(np.real(-c2*np.sqrt(phi1)-c4*np.sqrt(phi2)-c6*np.sqrt(phi3))) >= 1e-16:
  print('Wall boundary condition failure for b(z,t)')
  return 
 
 b = np.zeros([Nz,Nt])
 u = np.zeros([Nz,Nt])
 v = np.zeros([Nz,Nt])

 for i in range(0,Nz):
  for j in range(0,Nt):
   u[i,j] = np.real( ( u1*np.exp(-np.sqrt(phi1)*z[i]) + u2*np.exp(-np.sqrt(phi2)*z[i]) + \
                       u3*np.exp(-np.sqrt(phi3)*z[i]) + ap*omg ) * np.exp(1j*omg*t[j]) ) / (N**2.*np.sin(tht))

   v[i,j] = np.real( ( v1*np.exp(-np.sqrt(phi1)*z[i]) + v2*np.exp(-np.sqrt(phi2)*z[i]) + \
                       v3*np.exp(-np.sqrt(phi3)*z[i]) + ap*(omg**2.-(N*np.sin(tht))**2 ) - \
                       A*N**2.*np.sin(tht) ) * 1j * np.exp(1j*omg*t[j]) ) / (f * np.cos(tht) * N**2.*np.sin(tht)) 

   b[i,j] = np.real( ( c2*np.exp(-np.sqrt(phi1)*z[i]) + c4*np.exp(-np.sqrt(phi2)*z[i]) + \
                       c6*np.exp(-np.sqrt(phi3)*z[i]) + ap ) * 1j * np.exp(1j*omg*t[j]) ) 

 return b, u, v






