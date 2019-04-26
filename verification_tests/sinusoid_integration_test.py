# Simple RK4 verification
# Bryan Kaiser
# / /2019


import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/path/to/application/app/folder')
import functions as fn

#figure_path = "./figures/"
figure_path = './verification_tests/figures/sinusoid_integration_test/'

# Summary:
# a) division by zero can be a problem when 1/omg is part of the solution
# b) there's something wrong when time = [0,T] 
# while omg*t = [0,2pi]. The solution is to integrate forward in 
# radians so both are [0,2pi].


# =============================================================================


# non-dimensional time / frequency
T = 2.*np.pi #10. # s, period
omg = 1. #2.*np.pi/T
params = {'omg':omg}
Phi0 = 0.5
soln1 = Phi0*np.exp(np.sin(2.*np.pi)/omg) # analytical solution
Phin1,final_time = fn.rk4_time_step( params, Phi0 , T/2000, T , 'forcing_test' )
"""
print('final time =', final_time)
print('final radians =', final_time*omg)
print('Nt = ',2000)
print('Analytical solution = ',soln1)
print('Computed solution = ',Phin1)
"""

# dimensional time 
T = 10. # s, period
omg = 2.*np.pi/T
params = {'omg':omg}
Phi0 = 0.5
soln = Phi0*np.exp(np.sin(2.*np.pi)/omg)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/2000, T , 'forcing_test' )
"""
print('\nfinal time =', final_time)
print('final radians =', final_time*omg)
print('Nt = ',2000)
print('Analytical solution = ',soln)
print('Computed solution = ',Phin)
"""

# dimensional time:
T = 100. # s, period
omg = 2.*np.pi/T
params = {'omg':omg}
Phi0 = 0.5
soln = Phi0*np.exp(np.sin(2.*np.pi)/omg)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/2000, T , 'forcing_test' )
"""
print('\nfinal time =', final_time)
print('final radians =', final_time*omg)
print('Nt = ',2000)
print('Analytical solution = ',soln)
print('Computed solution = ',Phin)
"""

# this illustrates that as omg -> 0, holding Nt constant, the solution becomes 
# more numerically unstable, unless you choose to non-dimensionalize the time.
# you have to increase the number of time steps dramatically to 
# get to the same answers:

"""
# dimensional time 
T = 10. # s, period
omg = 2.*np.pi/T
params = {'omg':omg}
Phi0 = 0.5
soln = Phi0*np.exp(np.sin(2.*np.pi)/omg)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/20000, T , 'forcing_test' )
print('\nfinal time =', final_time)
print('final radians =', final_time*omg)
print('Nt = ',20000)
print('Analytical solution = ',soln)
print('Computed solution = ',Phin)

# dimensional time:
T = 100. # s, period
omg = 2.*np.pi/T
params = {'omg':omg}
Phi0 = 0.5
soln = Phi0*np.exp(np.sin(2.*np.pi)/omg)
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/200000, T , 'forcing_test' )
print('\nfinal time =', final_time)
print('final radians =', final_time*omg)
print('Nt = ',200000)
print('Analytical solution = ',soln)
print('Computed solution = ',Phin)
"""

# this further illustrates that turning up the number of time steps doesn't 
# help the fact that the analytical solution involves dividing by zero.
# Summary: the error grows as omega->0, even for computing the analytical 
# solution. 

omg = np.logspace(-18,0,num=19,endpoint=True)
#print(omg)


plotname = figure_path +'divide_by_zero_error.png' 
plottitle = r"numerical error as $\omega\rightarrow0$"
fig = plt.figure()
plt.loglog(omg,(abs(soln1-Phi0*np.exp(np.sin(2.*np.pi)/omg))/abs(soln1))*100.,'b',label=r"RK4")
plt.loglog(2.*np.pi/44700.*np.ones([19]),(abs(soln1-Phi0*np.exp(np.sin(2.*np.pi)/omg))/abs(soln1))*100.,'k')
plt.xlabel(r"$\omega$",fontsize=13);
plt.ylabel(r"% error",fontsize=13); 
plt.legend(loc=1,fontsize=14); 
plt.title(plottitle);
plt.grid()
plt.savefig(plotname,format="png"); plt.close(fig);

check_flag = 0
#print(abs(soln1-Phin1)/abs(soln1))
if abs(soln1-Phin1)/abs(soln1) < 1e-13:
  check_flag = check_flag + 1


# second test: solve log(u)=int(t*exp(-z/d)*cos(t-z/d))dt from 0 to 2pi
#print('\nTest 2:\n')

# only works for T in radians. Why? omg*timed is the same either way, the problem arises in the time stepper.
T = 2.*np.pi # s, period
omg = 2.*np.pi/T
nu = 1e-6
Re = 400.
dS = np.sqrt(2.*nu/omg) # Stokes' 2nd problem BL thickness
#print(T,omg,dS)
U = Re * (nu/dS)
Hd = 350.*dS # m, dimensional domain height (arbitrary choice)
Nz = 100 # number of grid points
z,dz = fn.grid_choice( 'cosine' , Nz , 1. ) # non-dimensional grid
#print('non-dimensional grid min/max: ',np.amin(z*Hd/dS),np.amax(z*Hd/dS))
#print(z[3]*Hd/dS)
#print()
grid_flag = 'cosine' # 'uniform' 
wall_flag = 'moving' 
params = {'nu': nu, 'omg': omg, 'T': T, 'Td':T, 'U': U,  
          'Hd': Hd, 'Nz':Nz, 'Re':Re,
          'dS':dS, 'z':z, 'dz':dz}

Phi0 = 1. #0.5
#if omg == 1.:
#soln2 = Phi0*np.exp(-2.*np.pi*np.exp(-z[3]*Hd/dS)*np.sin(z[3]*Hd/dS)) # analytical solution for omega=1
#else:
soln2 = Phi0*np.exp( np.exp(-z[3]*Hd/dS)*( 2.*np.pi*omg*np.sin(2.*np.pi*omg-z[3]*Hd/dS)+np.cos(2.*np.pi*omg-z[3]*Hd/dS)-np.cos(z[3]*Hd/dS) ) / omg**2. )

#print(Phi0*np.exp(-2.*np.pi*np.exp(-z[4]*Hd/dS)*np.cos(z[4]*Hd/dS)))
#print(Phi0*np.exp(-2.*np.pi*np.exp(-z[2]*Hd/dS)*np.cos(z[2]*Hd/dS)))
Phin2,final_time = fn.rk4_time_step( params, Phi0 , T/1000, T , 'forcing_test2' )

"""
print('final time =', final_time)
print('final radians =', final_time*omg)
print('Nt = ',1000)
print('Analytical solution = ',soln2)
print('Computed solution = ',Phin2)
"""

# Test 2 has the division by zero problem as well, plus theres something wrong when time goes [0,T] 
# while omg*t goes [0,2pi]. This points to dt as the problem...why do I have to time step in radians??

#print(abs(soln2-Phin2)/abs(soln2))

if abs(soln2-Phin2)/abs(soln2) < 1.1e-10:
  check_flag = check_flag + 1



if check_flag == 2:
    print('\n :) Two Runge-Kutta radian time step criterions satisfied\n')
else:
    print('\n ERROR: At least one Runge-Kutta radian time step criterions not satisfied\n')


