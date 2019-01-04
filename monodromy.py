# Monodromy matrix 
# Bryan Kaiser
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

figure_path = "./monodromy/figures"
data_path = "./monodromy/data"


# Need to implement boundary conditions!!!
# Do that by setting some parts of submatrices to zero ... set them after 
# advancing?

# add chebyshev node discretization

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
 return La_inv

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

def make_stationary_matrices(dz,Nz,C,K2,tht,k):
 La_inv = make_Lap_inv(dz,Nz,K2)
 pz = make_partial_z(dz,Nz)
 r = make_r(dz,Nz,C,tht,k)
 P4 = np.dot(La_inv,r)
 return La_inv, pz, P4


def make_DI(dz,Nz,U,k,Re):
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

def make_D4(dz,Nz,U,k,Re,Pr):
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

def rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,time,Phi): 
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

# =============================================================================
# flow parameters

# fluid properties
N = 1e-3 # 1/s, buoyancy frequency
nu = 2.0e-6 # m^2/s, kinematic viscosity
Pr = 1. # Prandtl number

# forcing
T = 44700.0 # s, M2 tide period
Uw = 0.01 # m/s, oscillation velocity amplitude
C = 0.25 # N*sin(tht)/omg

# Chebyshev grid:
Nz = int(500) # number of points in the vertical
H = 4.

# try Nz=500 (dz=0.008), Nt=447000 (dt=0.1)

Nt = int(447000)
tp = np.linspace(0.0, T, num=Nt) # s
dt = tp[1] - tp[0] # s

L = Uw/(2.*np.pi/T) # L = L_excursion
k = 10.*2.*np.pi/L # perturbation wavenumber
l = 10.*2.*np.pi/L
K2 = k**2.+l**2.


(coeffs, z, dz, U, Uz, B, Bz) = inst_solutions(N,T,Uw,nu,Pr,C,0.,Nz,H)
#(coeffs, z, dz, U, Uz, B, Bz) = plug_solutions(N,T,Uw,nu,Pr,C,0.,Nz,H) # <-------------------------|||
omg = coeffs[1]
tht = coeffs[2]
thtcrit = coeffs[3]
Bw = coeffs[5]
cottht = np.cos(tht)/np.sin(tht)
Re = Uw**2./(nu*omg)

"""
# mean flow solutions 
(coeffs, z, dz, t, dt, U, Uz, B, Bz) =  solutions(N,T,Uw,nu,Pr,C,Nt,Nz,H)
omg = coeffs[1]
tht = coeffs[2]
thtcrit = coeffs[3]
Bw = coeffs[5]
cottht = np.cos(tht)/np.sin(tht)
"""

"""
# check mean flow with a plot of U
zd = Uw/N # d0sb
zlabel = 'z*N/U'
figure_name = '/u_subcrit.png'
fig = plt.figure()
plotname = figure_path + figure_name 
plot_title = 'subcritical, C = %.2f' %(C)
fig = plt.figure() 
plt.semilogy(U[:,0]/Uw,z/zd,'m',label='0'); 
plt.semilogy(U[:,int(Nt/4)]/Uw,z/zd,'r',label='T/4'); 
plt.semilogy(U[:,int(Nt/2)]/Uw,z/zd,'g',label='T/2'); 
plt.semilogy(U[:,int(3*Nt/4)]/Uw,z/zd,'b',label='3T/4'); 
plt.semilogy(U[:,Nt-1]/Uw,z/zd,'--k',label='T'); 
plt.xlabel('u/U'); plt.legend(loc=4); 
plt.ylabel(zlabel); 
plt.title(plot_title);
plt.axis([-1.2,1.2,3e-6,10.]) #'tight')  
plt.grid()
plt.savefig(plotname,format="png"); 
plt.close(fig)
"""

 
# initialization:

# initialized submatrices of "A"
DI = np.zeros([Nz,Nz], dtype=complex)
D4 = np.zeros([Nz,Nz], dtype=complex)
La = np.zeros([Nz,Nz], dtype=complex)
q = np.zeros([Nz,Nz], dtype=complex)
r = np.zeros([Nz,Nz], dtype=complex)
pz = np.zeros([Nz,Nz], dtype=complex)

# stationary submatrices of "A" (do this once!)
(La_inv, pz, P4) = make_stationary_matrices(dz,Nz,C,K2,tht,k) 

if np.any(np.isnan(La_inv)):
 print('NaN detected in La_inv')
if np.any(np.isinf(La_inv)):
 print('Inf detected in La_inv')
if np.any(np.isnan(pz)):
 print('NaN detected in pz')
if np.any(np.isinf(pz)):
 print('Inf detected in pz')
if np.any(np.isnan(P4)):
 print('NaN detected in P4')
if np.any(np.isinf(P4)):
 print('Inf detected in P4')


# fundamental solution matrix
Phin = np.identity(int(4*Nz), dtype=complex) 

# time advancement:
for n in range(0,Nt):

 time = tp[n]
 print(time/T)

 # Runge-Kutta, 4th order:
 Phi1 = Phin; t1 = time; 
 k1 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t1,Phi1)
 del Phi1
 
 Phi2 = Phin + k1*(dt/2.); t2 = time + dt/2.;
 k2 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t2,Phi2)
 del Phi2

 Phi3 = Phin + k2*(dt/2.); t3 = time + dt/2.; 
 k3 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t3,Phi3)
 del Phi3

 print(k3)
 """
 if np.any(np.isnan(k3)):
  print('NaN detected in k3')
  break
 if np.any(np.isinf(k3)):
  print('NaN detected in k3')
  break
 """

 Phi4 = Phin + k3*dt; t4 = time + dt; 
 k4 = rk4(La_inv,pz,P4,k,l,N,T,Uw,nu,Pr,C,Nz,H,t4,Phi4)
 del Phi4

 Phin = Phin + ( k1 + k2*2. + k3*2. + k4 )*dt/6.;

 if np.any(np.isnan(Phin)):
  print('NaN detected in Phi')
  break
 if np.any(np.isinf(Phin)):
  print('NaN detected in Phi')
  break

# eigenvalues, eigenvectors
#(eigval, eigvec) = np.linalg.eig(Phin) 
eigvalues = np.linalg.eigvals(Phin) 

# Floquet exponents
flex = np.log(eigvalues)  
if np.any(np.isnan(flex)):
 print('NaN detected in flex')
if np.any(np.isinf(flex)):
 print('NaN detected in flex')
#print(flex)

# save:
savename = data_path + 'floquet_Re%i_C%i_Pr%i.h5' %(int(Re),int(C),int(Pr)) #'/circulation_%i_%i.h5' %(N0,N1) 
f2 = h5py.File(savename, "w")
dset = f2.create_dataset('T', data=T, dtype='f8')
dset = f2.create_dataset('t', data=tp, dtype='f8') 
dset = f2.create_dataset('dt', data=dt, dtype='f8') 
dset = f2.create_dataset('z', data=z, dtype='f8')
dset = f2.create_dataset('dz', data=dz, dtype='f8')
dset = f2.create_dataset('H', data=H, dtype='f8')
dset = f2.create_dataset('Re', data=Re, dtype='f8')
dset = f2.create_dataset('Pr', data=Pr, dtype='f8')
dset = f2.create_dataset('C', data=C, dtype='f8')
dset = f2.create_dataset('nu', data=nu, dtype='f8')
dset = f2.create_dataset('omg', data=omg, dtype='f8')
dset = f2.create_dataset('U', data=Uw, dtype='f8')
dset = f2.create_dataset('N', data=N, dtype='f8')
dset = f2.create_dataset('tht', data=tht, dtype='f8')
#dset = f2.create_dataset('eigval', data=eigval, dtype='f8')
#dset = f2.create_dataset('eigvec', data=eigvec, dtype='f8')
dset = f2.create_dataset('eigvalues', data=eigvalues, dtype='f8')
dset = f2.create_dataset('flex', data=flex, dtype='f8')
print('\nFloquet multiplier computed and written to file' + savename + '.\n')





# clean up
# add loops over what?
# now a propogator save file

 
# Now do eigenvalue analysis to get the eigenvalues of Phi(T). 
# these eigenvalues are the Floquet multipliers, where the multipliers are defined 
# by the Floquet modes: v(T) = multiplier * v(0). The Floquet exponents 
# are defined as multiplier = exp(sig + i*eta), where sig is the real part of 
# the Floquet exponent.
# if the real part of the Floquet exponent is positive then the Floquet 
# mode grows. The imaginary part of the Floquet exponent influences the frequency of the Floquet mode.


# log(multiplier) = exponent*T
