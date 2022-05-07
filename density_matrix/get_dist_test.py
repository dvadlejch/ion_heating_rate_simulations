# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:40:07 2019

@author: dv3
"""
import numpy as np
def get_distribution(n_avg, epsilon):
    N = 1
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

n_avg = 500
epsilon = 1e-3

N, n, p_n, error_prob = get_distribution(n_avg, epsilon)