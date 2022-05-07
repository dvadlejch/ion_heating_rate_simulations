# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:22:55 2019

@author: dv3
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:25:13 2019

@author: dv3
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy import constants
#from os import sys
from rabiflop_fit_multiple_laser import rabiflop,rabiflopfit
from scipy.optimize import least_squares, curve_fit

#==============data=============
data = pd.read_excel('flop_test.xlsx') # reads values from excel where the first column contains probe duration in [ms] and other
# columns contain excitation prob.
data = data.values

delay = data[0,1:]*1e-3
dur = data[1:,0]* 1e-3
prob_matrix = data[1:,1:]*1e-2
#----------applying binomial distribution--------------
n = 16
weight_matrix_data = ( prob_matrix*(1-prob_matrix)/n )**-1
nanval = np.logical_not(np.isnan(prob_matrix)) # array of bools containing info. about which of elements from prob_mat are nan
#==============importing data======

#==========just for plotting=========
points_per_flop = 30
maxflops = 2
#============

#............I need to define fitting function which fits inputted data and returns <n> and pi_pulse..............
def fitting(dur, prob, weight):
    # dur......probe times coresponding to measured data
    # prob.....measured excitation probabilities
    # delay_time..... time elapsed between stop cooling and start of probing 
    n_avg = 50  # initial point of the fitting
    pi_pulse = 50 * 1e-3 # initial point of the fitting
    decoh = 200*1e-3 # initial decoherence time
    
    
    #==========here I'm defining a function which is returning residuums squared==========
    def resid(a, dur, prob, weight):
        return  np.sqrt(weight)*( prob - rabiflopfit(a[0], a[1], a[2], dur) )
    
    a0 = [n_avg , pi_pulse, decoh] # initial point of the fitting
    optim = least_squares(resid, a0, args=(dur, prob, weight), xtol=1e-10, bounds=([0, np.inf]), loss='soft_l1') # bounds should be set in a way that <n> and pi_pulse can only be positive
    par = optim.x
    
    #========for unc. estimation========
    jac = optim.jac # jacobi matrix
    Ca = np.linalg.inv( np.matmul( np.transpose(jac), np.matmul( np.diag(weight), jac ) ) )# variance-covariance matrix for all parameters
    #residual_sq = optim.fun**2
    #g_fit = residual_sq.sum() * 1/(len(dur) - 2) # goodness-of-fit evaluation according to T.Strutz: Data Fitting p.114
    #===================================
    
    n_avg_o = par[0]
    pi_pulse_o = par[1]
    decoh_o = par[2]
    n_avg_unc =  np.sqrt(Ca[0,0])
    pi_pulse_unc =  np.sqrt(Ca[1,1])
    decoh_unc = np.sqrt(Ca[2,2])
    return n_avg_o, pi_pulse_o, n_avg_unc, pi_pulse_unc, decoh_o, decoh_unc
#................................................................................................................

#...........========now I want to calculate <n> and pi pulse for each data set also with uncertainty========...........
n_avg = np.zeros(len(delay),)
pi_pulse = np.zeros(len(delay),)
decoh = np.zeros(len(delay),)
decoh_u = np.zeros(len(delay),)
n_avg_unc = np.zeros(len(delay),)
pi_pulse_unc = np.zeros(len(delay),)
fig1 = plt.figure() # for plotting in the loop below
plt.rc('mathtext', fontset='cm')

error_filter = [] # list of bools for getting rid of <n> which doesnt make sence 

for i in range(len(delay)):
    prob = prob_matrix[ nanval[:,i],i ] # takes vector of prob. from prob. matrix for each column
    weight_prob = weight_matrix_data[ nanval[:,i],i ] # weight vector coresponding to prob data taken
    dur_nan = dur[ :len(prob) ]
    n_avg[i], pi_pulse[i], n_avg_unc[i], pi_pulse_unc[i], decoh[i], decoh_u[i] = fitting(dur_nan, prob, weight_prob) 
    t, c12, error = rabiflop(n_avg[i], pi_pulse[i], decoh[i], points_per_flop, maxflops )
    
    #====making subplot
    ax1 = fig1.add_subplot(2, 4, i+1)
    ax1.errorbar(dur_nan*1000, prob*100, yerr=np.sqrt( weight_prob**-1 )*100, fmt='.', linewidth= 1.5)
    ax1.plot(t*1000,c12*100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    title = '<n> = ' + format(n_avg[i], '0.2f') + r'$\pm$' + format(n_avg_unc[i], '0.1f')
    ax1.set_title(title, fontsize=14)
    plt.xlabel('t [ms]', fontsize=14)
    plt.ylabel('prob. [%]', fontsize=14)
    #=======
    
    if error > 0.05:
        print('Warning: error caused by finite summation limit insted of infinite is greater than 5% for fit #',i+1,'  error estimate = ',error)
        
        #text1 = r'error $\sim$'+ format(error, '0.3f')
        #ax1.text(150,26,text1, fontsize=14)
    else:
        error_filter.append(i)
#............................................................................................................................
        
#.........now I'm fitting <n> vs delay time.......................................        
n_avg = n_avg[error_filter] # choice of only meaningful <n>
delay = delay[error_filter] # similar to above
#weights2 = np.diag(1/n_avg_unc**2) # weights matrix for <n>......1/sigma^2

#================fitting of <n>(delay) function===============
def linfunc(x,a,b):
    return a*x + b
coefs, pcov = curve_fit(linfunc, delay, n_avg, sigma=n_avg_unc[error_filter], absolute_sigma=True) # absolute_sigma false or true?

#=====heating rate evaluation===========
heating_rate = coefs[0]
perr = np.sqrt(np.diag(pcov)) # estimated error
heating_rate_unc = perr[0]
#................................................................................

#==========vectors for plotting fitted line=====================
X = np.linspace(delay.min(), delay.max(), 200)
Y = linfunc(X,heating_rate,coefs[1])

#=============errorbar plot here ===============================
fig2 = plt.figure()
fig2.gca().errorbar(delay*1000, n_avg, yerr=n_avg_unc[error_filter], fmt='.', linewidth= 2, markersize=10 )
fig2.gca().plot(X*1000, Y)
plt.xlabel('delay [ms]', fontsize=14)
plt.ylabel('<n> [-]', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig2.gca().grid()
text_result = r'$\rm{d}<n>  / \rm{d}t  = $'+ format(heating_rate, '0.1f') + r'$\pm$ ' + format(heating_rate_unc,'0.1f') +' [1/s]' 
fig2.gca().set_title(text_result, fontsize=18)
#===============================================================

plt.show()
print('heating rate = ',heating_rate)
#plt.figure(1)
#plt.plot(t,c12)
#plt.plot(dur, prob_data, '.')
#plt.show()

#print('<n> = ',n_avg_o)
#print('pi pulse = ',1e3*pi_pulse_o,' [ms]')
#print('heating rate = ',heat_rate,' [1/s]')
