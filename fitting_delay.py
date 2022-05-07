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
from rabiflop_fit import rabiflop,rabiflopfit
from scipy.optimize import least_squares

#==============data=============
data = pd.read_excel('flop_test - Copy.xlsx') # reads values from excel where the first column contains probe duration in [ms] and other
# columns contain excitation prob.
data = data.values
dur = data[:,0]* 1e-3
prob = data[:,1]*1e-2
#delay_m10 = 1e-2*data[:,1]
#delay_m15 = 1e-2*data[:,2]
#delay_m30 = 1e-2*data[:,3]
#delay_m100 = 1e-2*data[:,4]
#delay_m100_2 = 1e-2*data[:,5]
#delay_m200 = 1e-2*data[:,6]
#delay_m500 = 1e-2*data[:,7]
#delay_m500 = delay_m500[np.logical_not(np.isnan(delay_m500))]
#dur = dur[:len(delay_m500)]
#==============importing data======


# ===============user input 
n_avg = 16.36  # initial point of the fitting
pi_pulse = 480 * 1e-3 # initial point of the fitting
delay_time = 0  # value from measurement
prob_data = prob # here I can change data which are to be fitted

#=====this is just for plotting====
points_per_flop = 20
maxflops = 2
#===================================

#==========here I'm defining a function which is returning residuums squared==========
def resid(a, dur, prob_data, delay_time):
    return  prob_data - rabiflopfit(a[0], a[1], delay_time, dur)

a0 = [n_avg , pi_pulse] # initial point of the fitting
optim = least_squares(resid, a0, args=(dur, prob_data, delay_time), xtol=1e-10, bounds=([0, np.inf])) # bounds should be set in a way that <n> and pi_pulse can only be positive
par = optim.x

#========for unc. estimation========
jac = optim.jac # jacobi matrix
Ca = np.linalg.inv( np.matmul( np.transpose(jac), jac ) ) # variance-covariance matrix for all parameters
residual_sq = optim.fun**2
g_fit = residual_sq.sum() * 1/(len(dur) - 2) # goodness-of-fit evaluation according to T.Strutz: Data Fitting p.114
#===================================

n_avg_o = par[0]
pi_pulse_o = par[1]
n_avg_unc = np.sqrt(g_fit*Ca[0,0])
pi_pulse_unc = np.sqrt(g_fit*Ca[1,1])

#========rabiflop function returns fitted function in specified times for plotting
t, c12, heat_rate, error = rabiflop(n_avg_o, pi_pulse_o, delay_time, points_per_flop, maxflops)


plt.figure(1)
plt.plot(t,c12)
plt.plot(dur, prob_data, '.')
plt.show()

print('<n> = ',n_avg_o)
print('pi pulse = ',1e3*pi_pulse_o,' [ms]')
print('heating rate = ',heat_rate,' [1/s]')
