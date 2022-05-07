# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:38:42 2019

@author: dv3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp
import functions_file_real as func
import time
from scipy import constants

#--------input params
time0 = time.time()

#------initial temperature is assumed as Doppler cooling limit ----
T = 0.47e-3 # doppler cooling limit
omega_sec = 2*np.pi*463e3  #secular freq
n_avg0 = 1/(np.exp(constants.hbar*omega_sec/(constants.k*T)) - 1) # doppler limit
n_avg0 = 1
heating_rate = 100
epsilon = 1e-5
prob_scale = 1e6 # probability scaling factor
pi_pulse = 390.18e-3 #[ms]
maxflops = 5

#------ solve for time interval-
number_of_time_steps = 5000
t0 = 0
tf = 2*maxflops*pi_pulse
dt = (tf-t0)/number_of_time_steps
#-------------

Omega_strength = np.pi/(2*pi_pulse)
lamb = 467e-9 #E3 laser
#lamb = 436e-9 #E2 laser
theta = np.pi/2
m = 170.936*constants.value("atomic mass constant")    #=====relative atomic mass is 171?
hbar = constants.hbar
lamb_dicke = np.sqrt(hbar/(2*m*omega_sec)) * 2*np.pi/lamb * np.sin(theta)
#----------------------------------

#-----------function return probabilities of being in vibrational state n
N, n, p_n, error_prob = func.get_distribution(n_avg0, heating_rate, tf, epsilon)
p_n_final, n_avg_final = func.get_distribution(n_avg0, heating_rate, tf, epsilon, ret_p_n_final=True, N=N)
#---------now I need initial coefficients a_nk
a_nk = np.eye(N,2*N)
# |psi_n> coresponds to pure state of superposition of |g> and |e> in the nth vibrational level ---- ie. |psi_n> = a_nk * |u_k>
# a_nk is amplitude for |psi_n> to be in state |u_k>
#----------a_nk = delta_nk, which coresponds to 
            
            
#============================ initial rho vector =========================================
p_n *= prob_scale
rho_initial = func.get_rho_ground(N, a_nk, p_n) # initial rho vector
#=============function returns vector of Omega_n=================  
Omega_n = func.get_rabi_freq(N, lamb_dicke, Omega_strength)
#----------------------------------------------------
#======================================================================

#------------integrating ODE---------------------------------
def rho_changerate_vector(t, rho, Omega_n, N, n_avg0, heating_rate):
    dp_k_dt = func.der_prob_a_per_pn(n_avg0, heating_rate, N, t)
    drho = np.zeros(3*N)
    #print(dp_k_dt*rho[:N])
    drho[:N] = dp_k_dt*rho[:N] + 2*Omega_n*rho[2*N:] # ground states
    drho[N:2*N] = dp_k_dt*rho[N:2*N] - 2*Omega_n*rho[2*N:] # excited states
    drho[2*N:] = dp_k_dt*rho[2*N:] + Omega_n*(rho[N:2*N] - rho[:N]) # here I discard imaginary unit--- check!!!
    #print(np.linalg.norm(dp_k_dt*rho[:N])/np.linalg.norm(2*Omega_n*rho[2*N:]))
    #print(t)
    return drho
   
# Call the ODE solver
psoln = solve_ivp( lambda t,y: rho_changerate_vector(t, y, Omega_n=Omega_n, N=N, n_avg0=n_avg0, heating_rate=heating_rate ), (t0, tf), rho_initial, method='RK45', rtol=1e-14, atol=1e-14, max_step=dt)

#-------forming solution----------------
t = psoln.t
rho_sol = psoln.y.reshape(3*N, len(t))  #----rho_vector[k, time]
#--------------------------------------------------------------

#---- norm check-----
#---- check of everything during time--------
rho_norm = rho_sol[:2*N,:].sum(axis=0) * 1/prob_scale
#dp_k_dt_sum = np.zeros(len(t))
#dp_k_dt_sum_a = np.zeros(len(t))
#for k in range(len(t)):
#    dp_k_dt_sum[k] = func.der_prob_a_per_pn(n_avg0, heating_rate, N, t[k]).sum()
#    dp_k_dt_sum_a[k] = heating_rate*N*(N-1 - 2*(n_avg0 + heating_rate*t[k]))/(2*(n_avg0+heating_rate*t[k])*(1+n_avg0+heating_rate*t[k]))
#resid_dp_dt_sum = np.abs(dp_k_dt_sum - dp_k_dt_sum_a) * 1/prob_scale  


#--------calculating probability of being in |e> for each time------
expect_value_Pe = rho_sol[N:2*N,].sum(axis=0) * 1/prob_scale

#rho = func.rho_vec_to_mat(rho_sol[:,5],N)
#----------------------plot oscilations------------
plt.figure(1)
plt.plot(t,expect_value_Pe, label='density matrix code')
leg = plt.legend(loc='best', fancybox=True, fontsize=20)
#plt.figure(2)
#plt.plot(n,dp_k_dt[:,450],'.')

plt.show()
print(time.time() - time0)