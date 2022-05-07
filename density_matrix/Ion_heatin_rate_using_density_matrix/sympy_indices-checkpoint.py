# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:57:49 2019

@author: dv3
"""

from sympy import *
# Matrix size                                                                                                                                                                                 
h = symbols('h')
i = symbols('i')
N = symbols('N')

k = Dummy('k')
j = Dummy('j')
q = Dummy('q')
n = Dummy('n')

# Matrix elements                                                                                                                                                                             
rho = IndexedBase('R')
H = IndexedBase('H')
omega = IndexedBase('omeg')

# Indices                                                                                                                                                                                     
#k, j, q, n = map(tensor.Idx, ['k', 'j', 'q', 'n'])

H = h*omega[j]*KroneckerDelta(k,j+N+1) + h*omega[k]*KroneckerDelta(k, j-N-1)
# H[k,j]

suma = Sum( H.subs(j,q)*rho[q,j] - H.subs( [(k,j), (j,q)] )* rho[k,q], (q, 0, 2*N +1) )

summed = suma.doit()

#M = M0[r,s]
#Res = Sum(H*M, (r, 0, n) ).doit()
#print( Res.subs(p,0) )