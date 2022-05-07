# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:14:40 2019

@author: dv3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
#import pandas as pd
#from scipy import constants
#from os import sys
from rabiflop_fit_one_point import rabiflop,rabiflopfit
from scipy.optimize import least_squares
import uncertainties as un
import random as rd
#from matplotlib.pyplot import gca

#==============data=============

dur = np.array( [0, 0.48] )


weight_matrix_data = np.array( [1e6, 1/0.04**2] )
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
#------initial temperature is assumed as Doppler cooling limit ----
T = 0.47e-3 # doppler cooling limit
pi_pulse = 405 * 1e-3 # initial point of the fitting
heating_rate = 0.5 # initial heating rate [1/s]
omega_sec = 2*np.pi*470e3  #secular freq
n_avg = 1/(np.exp(constants.hbar*omega_sec/(constants.k*T)) - 1)

#=====this is just for plotting====
points_per_flop = 40
maxflops = 0.61
#===================================

#==========here I'm defining a function which is returning residuums squared==========
def resid(a,n_avg, pi_pulse, dur, prob_data, weight, omega_sec):
    return  np.sqrt(weight)*( prob_data - rabiflopfit(n_avg, pi_pulse, dur, omega_sec, a[0]) )



a0 = [heating_rate] # initial point of the fitting

N = 200
heating_rate_mat = np.zeros(N)
for i in range(N):
    prob_data = np.array( [1e-50, rd.normalvariate(0.77,0.04)] )
    optim = least_squares(resid, a0, args=(n_avg, pi_pulse, dur, prob_data, weight_matrix_data, omega_sec), xtol=1e-10, bounds=([0, np.inf])) # bounds should be set in a way that <n> and pi_pulse can only be positive
    par = optim.x
    
    #========for unc. estimation========
    #jac = optim.jac # jacobi matrix
    #Ca = np.linalg.inv( np.matmul( np.transpose(jac), np.matmul( np.diag(weight_matrix_data), jac ) ) )# variance-covariance matrix for all parameters
    #residual_sq = optim.fun**2
    #g_fit = residual_sq.sum() * 1/(len(dur) - 2) # goodness-of-fit evaluation according to T.Strutz: Data Fitting p.114
    #===================================
    
    heating_rate_mat[i] = par[0] 
    #heating_rate_u = np.sqrt(Ca[0,0])
    
    #heating_rate_unc = un.ufloat(heating_rate_o, np.sqrt(Ca[0,0]) )
    #========rabiflop function returns fitted function in specified times for plotting
    t, c12, n_avg_final, error, n_avg_t = rabiflop(n_avg, pi_pulse, points_per_flop, maxflops, omega_sec, par[0])
    
    plt.figure(1)
    plt.plot(t*1e3,c12*100)
    plt.plot(dur*1e3, prob_data*100, '.')
#========rabiflop function returns fitted function in specified times for plotting
#t, c12, n_avg_final, error, n_avg_t = rabiflop(n_avg, pi_pulse, points_per_flop, maxflops, omega_sec, heating_rate_o)




plt.figure(2)
plt.rc('mathtext', fontset='cm')
plt.hist(heating_rate_mat, 50, density=True)

#plt.plot(t*1e3,c12*100)
#plt.errorbar(dur*1e3, prob_data*100, yerr=100/(np.sqrt(weight_matrix_data)), fmt='.', linewidth= 2, markersize=10 )
#plt.xlabel(r'$t$ [ms]', fontsize=16)
#plt.ylabel(r'prob [%]', fontsize=16)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)



#plt.title(r'$\rm{d}<n>/\rm{d}t = $' + r'${:.3S}$'.format(heating_rate_unc), fontsize=20)
#ax1.text(150,26,text1, fontsize=14)

plt.show()

#print('heating rate = ',heating_rate_unc)