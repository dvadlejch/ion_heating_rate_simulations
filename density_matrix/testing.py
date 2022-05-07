# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:58:27 2019

@author: dv3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp
from functions_file import get_distribution, get_rabi_freq, get_exc_prob, get_a_nk, get_rho
import time

#--------input params
#N = 50 # number of vibrational levels taking into account - 1
#N += 1 # I want N coresponds to my convention on a paper --- same as substitution N -> N+1 ; number of vibrational levels taking into account
time0 = time.time()
n_avg = 0.5
epsilon = 1e-2
pi_pulse = 50e-3
maxflops = 2
t0 = 0
tf = 2*maxflops*pi_pulse
Omega_strength = np.pi/(2*pi_pulse)
omega_sec = 2*np.pi*470e3  #secular freq
lamb = 467e-9 #E3 laser
#lamb = 436e-9 #E2 laser
theta = np.pi/2
m = 170.936*constants.value("atomic mass constant")    #=====relative atomic mass is 171?
hbar = constants.hbar
lamb_dicke = np.sqrt(hbar/(2*m*omega_sec)) * 2*np.pi/lamb * np.sin(theta)
#----------------------------------

#-----------function return probabilities of being in vibrational state n
N, n, p_n, error_prob = get_distribution(n_avg, epsilon)

#---------now I need initial coefficients a_nk
a_nk = np.eye(N,2*N)
# |psi_n> coresponds to pure state of superposition of |g> and |e> in the nth vibrational level ---- ie. |psi_n> = a_nk * |u_k>
# a_nk is amplitude for |psi_n> to be in state |u_k>
#----------a_nk = delta_nk, which coresponds to 
            
            
#=====================================================================
#----------calculating of rho matrix
rho_kj = np.zeros([2*N,2*N])
for k in range(2*N):
    for j in range(2*N):
        summ = 0
        for i in range(N):
            summ = summ + a_nk[i,k]*a_nk[i,j]*p_n[i]
        rho_kj[k,j] = summ

rho_kj = rho_kj.astype(complex)# I need to use complex numbers latter 

#=============function returns vector of Omega_n=================  
Omega_n = get_rabi_freq(N, lamb_dicke, Omega_strength)
#----------------------------------------------------
#======================================================================


#----------------calculating hamiltonian----------------------
hamil = np.zeros([2*N,2*N])
for k in range(N):
    for j in range(2*N):
        if j == k + N:
            hamil[k,j] = hbar*Omega_n[k]
            hamil[j,k] = hamil[k,j]
#------------------------------------------------------------



#------------integrating ODE---------------------------------
#===================================================================================

def rho_changerate(t, rho, hamiltonian,N):
    comut = hamil @ rho.reshape(2*N,2*N) - rho.reshape(2*N,2*N) @ hamil 
    return -1/constants.hbar*1j*comut.flatten()

# Call the ODE solver
psoln = solve_ivp( lambda t,y: rho_changerate(t, y, hamiltonian=hamil, N=N ), (t0, tf), rho_kj.flatten(), method='RK45', rtol=1e-10, atol=1e-10 )

#-------forming solution----------------
t = psoln.t
rho_sol = psoln.y.reshape(2*N,2*N,len(t))  #----rho[ k,j, time_index ]
#--------------------------------------------------------------


##==================trying to calculate a_nk in each time==============
#a_nk_t = np.zeros( (N,2*N,len(t) ) ).astype(complex)
#for i in range(len(t)):
#    a_nk_t[:,:,i] = get_a_nk(rho_sol[:,:,i], p_n, N)
##===================================================================================
#--------looks like it works



#--------calculating probability of being in |e> for each time------
expect_value_Pe = get_exc_prob(rho_sol, t, N)




#proj_e = 1/N * np.matmul( np.mat( np.concatenate( (np.zeros(N), np.ones(N)), axis=0 ) ).T, np.mat( np.concatenate( (np.zeros(N), np.ones(N)) )) )
#expect_value_Pe = np.zeros(len(t))
#for i in range(len(t)):
#    expect_value_Pe[i] = np.trace( np.matmul(proj_e, rho_sol[:,:,i]) )
#------------expectation value of probability of atom being in state |e>-----------------------

#-------------test of sum of all elements of rho related to excited state
#sum_test = np.zeros(len(t))
#for i in range(len(t)):
#    sum_test[i] = rho_sol[N:,:,i].sum()


#----------------------plot oscilations------------
plt.figure(1)
plt.plot(t,expect_value_Pe)
#plt.figure(2)
#plt.plot(n,p_n/p_n[0],'.')

plt.show()
print(time.time() - time0)