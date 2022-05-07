# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:48:05 2019

@author: dv3
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:32:13 2019

@author: dv3
"""

#
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from scipy import constants
#from os import sys




#Rabi flopping parameters
def rabiflopfit(n_avg0, pi_pulse, heat_rate, delay_time, t):
    omega_sec = 2*np.pi*470e3  #secular freq
    lamb = 467e-9 #E3 laser
    #lamb = 436e-9 #E2 laser
    theta = np.pi/2
    #Omega_strength = 2*np.pi*50  #total coupling strength 
    m = 171*constants.value("atomic mass constant")    #=====relative atomic mass is 171?
    hbar = constants.hbar
    lamb_dicke = np.sqrt(hbar/(2*m*omega_sec)) * 2*np.pi/lamb * np.sin(theta) #Lamb-Dicke parametr r
    #n_avg = float(input('Initial <n> = '))
    #delay_time = float(input('Delay time [ms] = '))
    delay_time = delay_time*1e-3
    #pi_pulse = float(input('pi pulse [ms] = '))
    #pi_pulse = pi_pulse*1e-3
#    points_per_flop = 20  #time points per one flop
#    maxflops = float(input('How many flops do you want to see? maxflops = ')) #determines max. time of probing
    nmax = 1000 # upper summation limit
     
    Omega_strength = np.pi / pi_pulse
    #===========time array
#    tper = 2*np.pi/Omega_strength
#    tstep = tper/points_per_flop
#    t = np.arange(0,maxflops*tper,tstep)
    tlen = len(t)
    
    #============leg. poly evaluation
    leg = np.zeros( (nmax,) )
    leg[0] = 1
    leg[1] = 1 - lamb_dicke**2
    
    for k in range(1,nmax-1):
        leg[k+1] = 1/k*((2*(k-1)+1 - lamb_dicke**2)*leg[k]-(k-1)*leg[k-1])
    
    #==============================  
    Omega_n = Omega_strength * np.exp(-lamb_dicke**2/2) * leg  
    n = np.arange(0,nmax)
    n_avg0 += heat_rate*delay_time
    
    if heat_rate == 0:
        #======vibrational states distribution
        fr1 = 1/(1 + n_avg0)
        fr2 = n_avg0/(1+n_avg0) 
        p_n = fr1*fr2**n
        #error = np.abs(1-p_n.sum())
        #================================   
        #==========summation
        summ_nj = np.zeros((nmax,tlen))
        for j in range(tlen):
            summ_nj[:,j] = p_n* np.cos(Omega_n * t[j])
        
        summ = summ_nj.sum(axis=0)
        c12 = 0.5*(1 - summ)
        #========================
    else:  
        #==========summation
        summ_nj = np.zeros((nmax,tlen))
        for j in range(tlen):
            #======vibrational states distribution
            fr1 = 1/(1 + n_avg0)
            fr2 = n_avg0/(1+n_avg0) 
            p_n = fr1*fr2**n
            summ_nj[:,j] = p_n* np.cos(Omega_n * t[j])
            
            #======change of <n>
            if j == tlen-1:
                break
            else:
                n_avg0 += heat_rate* (t[j+1]-t[j])
        
        #error = np.abs(1 - p_n.sum())    
        summ = summ_nj.sum(axis=0)
        c12 = 0.5*(1 - summ)
        
    return c12

def rabiflop(n_avg, pi_pulse, heat_rate, delay_time, points_per_flop, maxflops):
    omega_sec = 2*np.pi*470e3  #secular freq
    lamb = 467e-9 #E3 laser
    #lamb = 436e-9 #E2 laser
    theta = np.pi/2
    #Omega_strength = 2*np.pi*50  #total coupling strength 
    m = 171*constants.value("atomic mass constant")    #=====relative atomic mass is 171?
    hbar = constants.hbar
    lamb_dicke = np.sqrt(hbar/(2*m*omega_sec)) * 2*np.pi/lamb * np.sin(theta) #Lamb-Dicke parametr r
    #n_avg = float(input('Initial <n> = '))
    #delay_time = float(input('Delay time [ms] = '))
    delay_time = delay_time*1e-3
    #pi_pulse = float(input('pi pulse [ms] = '))
    #pi_pulse = pi_pulse*1e-3
#    points_per_flop = 20  #time points per one flop
#    maxflops = float(input('How many flops do you want to see? maxflops = ')) #determines max. time of probing
    nmax = 1000 # upper summation limit
     
    Omega_strength = np.pi / pi_pulse
    #===========time array
    tper = 2*np.pi/Omega_strength
    tstep = tper/points_per_flop
    t = np.arange(0,maxflops*tper,tstep)
    tlen = len(t)
    
    #============leg. poly evaluation
    leg = np.zeros( (nmax,) )
    leg[0] = 1
    leg[1] = 1 - lamb_dicke**2
    
    for k in range(1,nmax-1):
        leg[k+1] = 1/k*((2*(k-1)+1 - lamb_dicke**2)*leg[k]-(k-1)*leg[k-1])
    
    #==============================  
    Omega_n = Omega_strength * np.exp(-lamb_dicke**2/2) * leg  
    n = np.arange(0,nmax)
    n_avg += heat_rate*delay_time
    
    if heat_rate == 0:
        #======vibrational states distribution
        fr1 = 1/(1 + n_avg)
        fr2 = n_avg/(1+n_avg) 
        p_n = fr1*fr2**n
        error = np.abs(1-p_n.sum())
        #================================   
        #==========summation
        summ_nj = np.zeros((nmax,tlen))
        for j in range(tlen):
            summ_nj[:,j] = p_n* np.cos(Omega_n * t[j])
        
        summ = summ_nj.sum(axis=0)
        c12 = 0.5*(1 - summ)
        #========================
    else:  
        #==========summation
        summ_nj = np.zeros((nmax,tlen))
        for j in range(tlen):
            #======vibrational states distribution
            fr1 = 1/(1 + n_avg)
            fr2 = n_avg/(1+n_avg) 
            p_n = fr1*fr2**n
            summ_nj[:,j] = p_n* np.cos(Omega_n * t[j])
            
            #======change of <n>
            n_avg += heat_rate*tstep
        
        error = np.abs(1 - p_n.sum())    
        summ = summ_nj.sum(axis=0)
        c12 = 0.5*(1 - summ)
    
    
    return t, c12, n_avg, error


