# 
# Bryan Kaiser
# 3/ /2019


import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functions as fn

figure_path = "./figures/"


# =============================================================================

T = 10. # s, period
omg = 2.*np.pi/T
params = {'omg':omg}

Phi0 = 0.5
Phin,final_time = fn.rk4_time_step( params, Phi0 , T/2000, T , 'forcing_test' )

print('final time =', final_time)
print('final radians =', final_time*omg)
print('Analytical solution = ',Phi0*np.exp(np.sin(omg*final_time)/omg))
print('Computed solution = ',Phin)


