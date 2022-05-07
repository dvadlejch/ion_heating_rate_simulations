# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:49:29 2019

@author: dv3
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:42:41 2019

@author: dv3
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy import constants
#from os import sys
from rabiflop_fit_heating import rabiflop,rabiflopfit
from scipy.optimize import least_squares

#==============data=============
data = pd.read_excel('flop_test - Copy.xlsx') # reads values from excel where the first column contains probe duration in [ms] and other
# columns contain excitation prob.
data = data.values
dur = data[:,0]* 1e-3
prob = data[:,1]*1e-2
weight = np.array( [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] )
#weight = np.ones(len(dur),)
#delay_m10 = 1e-2*data[:,1]
#delay_m15 = 1e-2*data[:,2]
#delay_m30 = 1e-2*data[:,3]
#delay_m100 = 1e-2*data[:,4]
#delay_m100_2 = 1e-2*data[:,5]
#delay_m200 = 1e-2*data[:,6]
#delay_m500 = 1e-2*data[:,7]
#delay_m500 = delay_m500[np.logical_not(np.isnan(delay_m500))]
#dur = dur[:len(delay_m500)]



# ===============user input 
n_avg0 = 50  # starting point
pi_pulse = 50 * 1e-3 # starting point
delay_time = 10  # must corespond to measured data
prob_data = prob
heat_rate = 10
points_per_flop = 30
maxflops = 1.5


def resid(a, n_avg0, dur, prob_data, delay_time, weight):
    return ( weight * ( prob_data - rabiflopfit(n_avg0, a[0], a[1], delay_time, dur) ) ) **2

a0 = [ pi_pulse, heat_rate]
optim = least_squares(resid, a0, args=(n_avg0, dur, prob_data, delay_time, weight), xtol=1e-10,  bounds=([0, np.inf]))
par = optim.x
pi_pulse_o = par[0]
heat_rate_o = par[1]

t, c12, n_avg_f, error = rabiflop(n_avg0, pi_pulse_o, heat_rate_o, delay_time, points_per_flop, maxflops)


plt.figure(1)
plt.plot(t*1e3,c12*100)
plt.plot(dur*1e3, prob_data*100, '.r')
plt.xlabel('pulse duration [ms]', fontsize=18)
plt.ylabel('jump probability [%]', fontsize=18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend( ['fit','data'] , loc='upper right', fontsize=18)
plt.tight_layout()
plt.grid()
plt.show()

print('<n> = ',n_avg_f)
print('pi pulse = ',1e3*pi_pulse_o,' [ms]')
print('heating rate = ',heat_rate_o,' [1/s]')
