# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:30:17 2019

@author: dv3
"""

import numpy as np
import mpmath as mpm

#----- function for estimating N needed for give precision ---
# input initial n_avg, heat rate, final time, precision, optional( N can be forced to some number )
def get_distribution(n_avg0, heating_rate, t_final, epsilon, set_N = False, N = 1, ret_p_n_final = False):
    if set_N == False and ret_p_n_final == False:
        n = np.arange(0,N)
        n_avg_final = n_avg0 + heating_rate*t_final
        fr1_f = 1/(1 + n_avg_final)
        fr2_f = n_avg_final/(1+n_avg_final)
        p_n_f = fr1_f*fr2_f**n
        error_prob = np.abs(1 - p_n_f.sum())
        while error_prob > epsilon:
            N += 1
            napp = np.arange(n.max()+1, N)
            n = np.append(n, napp)
            p_n_f = np.append(p_n_f, fr1_f*fr2_f**napp)
            error_prob = np.abs(1 - p_n_f.sum())
        
        fr1 = 1/(1 + n_avg0)
        fr2 = n_avg0/(1 + n_avg0)
        
        return N, n, fr1*fr2**n, error_prob  # prob. distribution of vibrational levels
    
    if set_N == True and ret_p_n_final == False:
        n = np.arange(0,N)
        fr1 = 1/(1 + n_avg0)
        fr2 = n_avg0/(1+n_avg0)
        p_n = fr1*fr2**n
        error_prob = np.abs(1 - p_n.sum())
        return N, n, fr1*fr2**n, error_prob
    
    if ret_p_n_final == True:
        n = np.arange(0,N)
        n_avg_final = n_avg0 + heating_rate*t_final
        fr1 = 1/(1 + n_avg_final)
        fr2 = n_avg_final/(1+n_avg_final)
        p_n = fr1*fr2**n
        error_prob = np.abs(1 - p_n.sum())
        return fr1*fr2**n, n_avg_final
    
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
    
    return rho_kj# I need to use complex numbers latter 
    
#----------calculating of initial rho-vector
# input: number of vibrational states, a_nk coeffs, probabilities
# output: rho vector: [diagonal elements coresponding to |g>, diag elements coresp. to |e>, off-diagonal elements]
def get_rho_ground(N, a_nk, p_n):
    rho = np.zeros(3*N)
    for k in range(N):
        rho[k] = p_n[k]*a_nk[k,k]**2
    for k in range(N,2*N):
        rho[k] = p_n[k-N]*a_nk[k-N,k]**2
    for k in range(2*N,3*N):
        rho[k] = p_n[k-2*N]*a_nk[k-2*N,k-2*N]*a_nk[k-2*N,k-N]
    
    return rho

def rho_vec_to_mat(rho_vec, N):
    rho = np.zeros([2*N,2*N]).astype(complex)
    rho[:N,:N] = np.diag(rho_vec[:N])
    rho[N:,N:] = np.diag(rho_vec[N:2*N])
    for i in range(2*N, 3*N):
        rho[i-2*N,i-3*N] = rho_vec[i]*1j
        rho[i-3*N, i-2*N] = -rho_vec[i]*1j
    return rho

#------------calculating Omega_n coupling constants  with better precision
def get_rabi_freq_mpmath(N, lamb_dicke, Omega_strength):
    mpm.mp.dps = 40
    leg = []
    leg.append(mpm.mpf(1))
    if N >=2:
        leg.append(mpm.mpf( 1 - lamb_dicke**2 ))
        for k in range(1,N-1):
            leg.append( mpm.mpf( 1/k*((2*(k-1)+1 - lamb_dicke**2)*leg[k]-(k-1)*leg[k-1]) ) ) 
    return np.array( [float(mpm.mpf( Omega_strength ) * mpm.exp(-lamb_dicke**2/2) * leg[i]) for i in range(len(leg)) ] )

#------------calculating Omega_n coupling constants 
def get_rabi_freq(N, lamb_dicke, Omega_strength):
    leg = np.zeros( N )
    leg[0] = 1
    if N >=2:
        leg[1] = 1 - lamb_dicke**2
        for k in range(1,N-1):
            leg[k+1] = 1/k*((2*(k-1)+1 - lamb_dicke**2)*leg[k]-(k-1)*leg[k-1])
    return Omega_strength * np.exp(-lamb_dicke**2/2) * leg 



    
#------------- time der. of prob. analytical ----------------
# input: n_average(t=0), heating rate, vibrational numbers, time
# output: vector of dp/dt, p_n(t)
def der_prob_a_per_pn(n_avg0, heating_rate, N, t):
    n = np.arange(0,N)
    n_avg = n_avg0 + heating_rate*t
    numer1 = heating_rate*(n_avg-n)
    denum1 = n_avg*(n_avg+1)
    return -numer1/denum1

#------------- time der. of prob. analytical ----------------
# input: n_average(t=0), heating rate, vibrational numbers, time
# output: vector of dp/dt, p_n(t)
def der_prob_a(n_avg0, scale, heating_rate, N, t):
    n = np.arange(0,N)
    n_avg = n_avg0 + heating_rate*t
    numer1 = n_avg - n
    numer2 = scale*heating_rate*( n_avg/(1+n_avg) )**(n-1)
    denum = (1+n_avg)**3
    fr1 = scale/(1 + n_avg)
    fr2 = n_avg/(1+n_avg)
    return -1*(numer1*numer2/denum), fr1*fr2**n
