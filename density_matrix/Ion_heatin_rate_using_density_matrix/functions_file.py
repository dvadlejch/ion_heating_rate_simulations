# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:00:13 2019

@author: dv3
"""
#-----------function return probabilities of being in vibrational state n
# function is seaching for optimum number of n taken into account according to given precision
import numpy as np
def get_distribution(n_avg, epsilon, set_N = False, N = 1):
    if set_N == False:
        n = np.arange(0,N)
        fr1 = 1/(1 + n_avg)
        fr2 = n_avg/(1+n_avg)
        p_n = fr1*fr2**n
        error_prob = np.abs(1 - p_n.sum())
        while error_prob > epsilon:
            N += 1
            napp = np.arange(n.max()+1, N)
            n = np.append(n, napp)
            p_n = np.append(p_n, fr1*fr2**napp)
            error_prob = np.abs(1 - p_n.sum())
        
        return N, n, p_n, error_prob  # prob. distribution of vibrational levels
    
    if set_N == True:
        n = np.arange(0,N)
        fr1 = 1/(1 + n_avg)
        fr2 = n_avg/(1+n_avg)
        p_n = fr1*fr2**n
        error_prob = np.abs(1 - p_n.sum())
        return N, n, fr1*fr2**n, error_prob
    
#---------------------------

#----------calculating of rho matrix
def get_rho(N, a_nk, p_n):
    rho_kj = np.zeros([2*N,2*N])
    for k in range(2*N):
        for j in range(2*N):
            summ = 0
            for i in range(N):
                summ = summ + a_nk[i,k]*a_nk[i,j]*p_n[i]
            rho_kj[k,j] = summ
    
    return rho_kj.astype(complex)# I need to use complex numbers latter 
    
#------------calculating Omega_n coupling constants 
def get_rabi_freq(N, lamb_dicke, Omega_strength):
    leg = np.zeros( N )
    leg[0] = 1
    if N >=2:
        leg[1] = 1 - lamb_dicke**2
        for k in range(1,N-1):
            leg[k+1] = 1/k*((2*(k-1)+1 - lamb_dicke**2)*leg[k]-(k-1)*leg[k-1])
    return Omega_strength * np.exp(-lamb_dicke**2/2) * leg 

#---------- function returns |e> probability for each time
# input: density matrix in each time, time vector, number of vibrational quanta
# output: probability vector
def get_exc_prob(rho_sol, t, N):
    expect_value_Pe = np.zeros(len(t))
    for i in range(len(t)):
        summ = 0
        for j in range(2*N):
            for k in range(2*N):
                if k==j and j>= N:
                    summ += rho_sol[j,k,i]
        expect_value_Pe[i] = summ
    return expect_value_Pe

#----------- function for calculating a_nk coefs for given rho
# input: density matrix, probability distribution of vibrational states, length of the vector of probabilities
# output: a_nk matrix
def get_a_nk(rho, p_n, N):
    a_nk = np.zeros( (N, 2*N) ).astype(complex)
    for n in range(N):
        if float(rho[n,n]) == 0:
            a_nk[n, n+N] = 1
        elif float(rho[n,n]) == p_n[n]:
            a_nk[n,n] = 1
        else:
#            a_nk[n,n] = rho[n,n+N]/(np.sqrt( p_n[n]*(p_n[n] - rho[n,n])) )
#            a_nk[n, n+N] = np.sqrt( 1 - rho[n,n]/p_n[n] )
            a_nk[n,n] = np.sqrt( rho[n,n]/p_n[n] )
            a_nk[n, n+N] = rho[n, n+N]* 1/np.sqrt(p_n[n]*rho[n,n])
    return a_nk
#-------------------------------------------------------------
    
#-------------- time der. of probabilities ------------------
# input: n_average(t), p_n(t), vibrational numbers vector, time step, heating rate
# output: dp/dp, n_average(t+dt), p_n(t+dt)
def der_prob(n_avg, p_n, n, dt, heating_rate):
    n_avg += heating_rate*dt
    fr1 = 1/(1 + n_avg)
    fr2 = n_avg/(1+n_avg)
    p_n_dt = fr1*fr2**n
    dp_n_dt = (p_n_dt - p_n)/dt
    return dp_n_dt, n_avg, p_n_dt
#------------------------------------------------------------
    
#------------- time der. of prob. analytical ----------------
# input: n_average(t=0), heating rate, vibrational numbers, time
# output: vector of dp/dt, p_n(t)
def der_prob_a(n_avg0, heating_rate, n, t):
    n_avg = n_avg0 + heating_rate*t
    numer1 = n_avg - n
    numer2 = heating_rate*( n_avg/(1+n_avg) )**(n-1)
    denum = (1+n_avg)**3
    fr1 = 1/(1 + n_avg)
    fr2 = n_avg/(1+n_avg)
    return -1*(numer1*numer2/denum), fr1*fr2**n
    
#------------- time der. of prob. analytical ----------------
# input: n_average(t=0), heating rate, vibrational numbers, time
# output: vector of dp/dt, p_n(t)
def der_prob_a_per_pn(n_avg0, heating_rate, N, t):
    n = np.arange(0,N)
    n_avg = n_avg0 + heating_rate*t
    numer1 = heating_rate*(n_avg-n)
    denum1 = n_avg*(n_avg+1)
    return -numer1/denum1



