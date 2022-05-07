# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:13:31 2019

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
#from matplotlib.pyplot import gca

#==============data=============

fig, ax1 = plt.subplots()
plt.rc('mathtext', fontset='cm')
colors = ['b', 'black', 'r']
j = 0
for point in [0.81, 0.77, 0.73]:
    
    dur = np.array( [0, 0.479, 0.481] )
    prob_data = np.array( [1e-50, point, point] )
    n = 16
    weight_matrix_data = np.array( [1e6, 1/0.04**2,  1/0.04**2] )
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
    pi_pulse = 400 * 1e-3 # initial point of the fitting
    heating_rate = 30 # initial heating rate [1/s]
    omega_sec = 2*np.pi*463e3  #secular freq
    n_avg = 1/(np.exp(constants.hbar*omega_sec/(constants.k*T)) - 1)
    
    #=====this is just for plotting====
    points_per_flop = 40
    maxflops = 0.7
    #===================================
    
    #==========here I'm defining a function which is returning residuums squared==========
    def resid(a,n_avg, dur, prob_data, weight, omega_sec):
        return  np.sqrt(weight)*( prob_data - rabiflopfit(n_avg, a[1], dur, omega_sec, a[0]) )
    
    
    
    a0 = [heating_rate, pi_pulse] # initial point of the fitting
    optim = least_squares(resid, a0, args=(n_avg, dur, prob_data, weight_matrix_data, omega_sec), xtol=1e-10, bounds=([0, np.inf])) # bounds should be set in a way that <n> and pi_pulse can only be positive
    par = optim.x
    
    #========for unc. estimation========
    jac = optim.jac # jacobi matrix
    Ca = np.linalg.inv( np.matmul( np.transpose(jac), np.matmul( np.diag(weight_matrix_data), jac ) ) )# variance-covariance matrix for all parameters
    #residual_sq = optim.fun**2
    #g_fit = residual_sq.sum() * 1/(len(dur) - 2) # goodness-of-fit evaluation according to T.Strutz: Data Fitting p.114
    #===================================
    
    heating_rate_o = par[0] 
    pi_pulse_o = par[1]
    
    #heating_rate_u = np.sqrt(Ca[0,0])
    
    #heating_rate_unc = un.ufloat(heating_rate_o, np.sqrt(Ca[0,0]) )
    
    
    #========rabiflop function returns fitted function in specified times for plotting
    t, c12, error, n_avg_t = rabiflop(n_avg, pi_pulse_o, points_per_flop, maxflops, omega_sec, heating_rate_o)
    
    
    
    
    
    
    
    ax1.plot(t*1e3,c12*100, colors[j], linewidth=2, label=r'$\frac{\rm{d}<n>}{\rm{d}t} = $'+ 	r'${:.1f}$'.format(heating_rate_o ) )
    
    
    j += 1
    #ax1.errorbar(dur*1e3, prob_data*100, yerr=100/(np.sqrt(weight_matrix_data)), fmt='.', linewidth= 2, markersize=10 )
    #ax1.plot(dur[2]*1e3, prob_data[2]*100, '.r', markersize=10)
    
#    ax2 = ax1.twiny()
#    ax2.plot(n_avg_t,c12*100)
#    ax2.tick_params(labelsize=20)
#    ax2.set_xlabel(r'$<n>$ [-]', fontsize=22)
#    ax2.grid(True)
    
#    ax1.set_axisbelow(True)
#   ax2.set_axisbelow(True)
    #ax1.set_xtickslabels(fontsize=14)
    #ax1.set_ytickslabels(fontsize=14)
    print('pi_pulse [s] = ',pi_pulse_o)
    print('heating rate = ',heating_rate_o)
    
    #plt.text(20,93,r'$\frac{\rm{d}<n>}{\rm{d}t} = $' + r'${:.3S}$'.format(heating_rate_unc), fontsize=30, color='r')
    #ax1.text(150,26,text1, fontsize=14)

ax1.set_xlabel(r'$t$ [ms]', fontsize=22)
ax1.set_ylabel(r'prob. [%]', fontsize=22)
ax1.tick_params(labelsize=20)
ax1.errorbar(dur[2]*1e3, 77, yerr=100/(np.sqrt(weight_matrix_data[2])), fmt='sg', linewidth= 2, markersize=12 )
leg = plt.legend(loc='best', fancybox=True, fontsize=20)
plt.title('Heating rate fit', fontsize=30)
ax2 = ax1.twiny()
ax2.plot(n_avg_t,np.zeros(len(n_avg_t)),'.w', markersize=0)
ax2.tick_params(labelsize=20)
ax2.set_xlabel(r'$<n>_{45.6}$ [-]', fontsize=22)
ax2.spines['top'].set_color('r')
ax2.grid(True)
ax1.yaxis.grid(True)

ax1.text(-10, 90, 'Assuming the worst case of secular freq. '+ r'$\omega_{\rm{s}} = $' + r'${:.0f}$'.format(omega_sec/(2*np.pi)/1e3) + ' [kHz]', fontsize=20)



#ax1.grid(True)

plt.show()

#print('heating rate = ',heating_rate_unc)