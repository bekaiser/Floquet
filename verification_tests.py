#
# Bryan Kaiser

# add all other tests
# check on discretization test case 3, 2nd derivative, lower BC
# make discretization test plots subplots
# check on base flow solutions for the moving wall.
# make sure the non-moving wall solutions are working as well (add stokes test to base flow test) 

import runpy
import os
import shutil
import numpy as np


# delete old plots:
folder = ('./verification_tests/figures/discretization_test/',
          './verification_tests/figures/GTE_test/',
          './verification_tests/figures/LTE_test/',
          './verification_tests/figures/sinusoid_integration_test/')

for j in range(0,4):
    for the_file in os.listdir(folder[j]):
        file_path = os.path.join(folder[j], the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)



# FIX: merely plot? check that the boundary conditions work? 
#print('Verifying that discrete derivatives are consistent with the laminar flow')
#runpy.run_path('./verification_tests/base_flow_test.py')

"""
print('\nVerifying the global truncation error curve for\n Runge-Kutta 4th-order time stepper:\n')
runpy.run_path('./verification_tests/GTE_test.py')

print('\nVerifying the local truncation error curve for\n Runge-Kutta 4th-order time stepper:\n')
runpy.run_path('./verification_tests/LTE_test.py')

print('\nVerifying Runge-Kutta radian time step test:\n')
runpy.run_path('./verification_tests/sinusoid_integration_test.py')

print('\nVerifying spatial discrete derivatives are computed properly:\n')
runpy.run_path('./verification_tests/discretization_test.py')

print('\nVerifying spatial discrete derivatives are computed properly:\n')
runpy.run_path('./verification_tests/discretization_test2.py')

print('\nVerifying changes to functions.py did not change Stokes 2nd problem stability:\n')
runpy.run_path('./verification_tests/stokes_check.py')

print('\nVerifying the non-dimensional base flow is :\n')
runpy.run_path('./verification_tests/base_flow_test2.py')
"""

print('\nVerifying Floquet stability calculation using the Mathieu equation:\n')
runpy.run_path('./verification_tests/Mathieu_test.py')



