# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:38:25 2019

@author: dv3
"""

import numpy as np
import matplotlib.pyplot as plt
#0from os import sys

from rabiflop_fit_one_point import rabiflop


# n_avg value, max. time etc.
n_avg = float(input('Initial <n> = '))
heat_rate = float(input('Heating rate [1/s] = '))
pi_pulse = float(input('pi pulse [ms] = '))
pi_pulse = pi_pulse*1e-3
points_per_flop = 100  #time points per one flop
omega_sec = 2*np.pi*470e3  #secular freq
maxflops = float(input('How many flops do you want to see? maxflops = ')) #determines max. time of probing
 
t, c12, error, n_avg_t = rabiflop(n_avg, pi_pulse, points_per_flop, maxflops, omega_sec, heat_rate)

plt.figure(1)
plt.plot(t,c12, label='old_code' )
plt.xlabel("t",fontsize=14)
plt.ylabel("c1^2",fontsize=14)
plt.grid()
plt.show()
#leg = plt.legend(loc='best', fancybox=True, fontsize=20)

if error > 1e-2:
    print("Warning: Error due to the finite upper limit of summation is higher than 1%!")
    print('Error estimate = ',error)
#choice = input('Do you want to run code again? (if yes press y): ')
#if choice == 'y':
#    main()
#else:
#    sys.exit()
        
