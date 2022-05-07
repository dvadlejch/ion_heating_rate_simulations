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
q = symbols('q')
n = Dummy('n')

# Matrix elements                                                                                                                                                                             
rho = IndexedBase('R')
rho_c = IndexedBase('R*')
rho_d = IndexedBase('R^.')
H = IndexedBase('H')
omega = IndexedBase('\Omega')
p = IndexedBase('p^.')
a = IndexedBase('a')
a_c = IndexedBase( 'a*' )
c_g = IndexedBase( 'c_g' )
c_e = IndexedBase( 'c_e' )
c_g_c = IndexedBase( 'c_g*' )
c_e_c = IndexedBase( 'c_e*' )

# Indices                                                                                                                                                                                     
#k, j, q, n = map(tensor.Idx, ['k', 'j', 'q', 'n'])

H = omega[j]*KroneckerDelta(k,j+N+1) + omega[k]*KroneckerDelta(k, j-N-1)
# H[k,j]

#suma = Sum( H.subs(j,q)*rho[q,j] - H.subs( [(k,j), (j,q)] )* rho[k,q], (q, 0, 2*N +1) )
suma1 = Sum( H.subs(j,q)*rho[q,j], (q, 0, 2*N +1) )
suma2 = Sum( H.subs( [(j,q), (k,j)] )* rho_c[q,k], (q, 0, 2*N+1 ) )
summed = suma1.doit() - suma2.doit() # hamiltonian part

#---- dpdt part
a = c_g[n]*KroneckerDelta(n,k) + c_e[n]*KroneckerDelta(n,k-N-1)
a_c = c_g_c[n]*KroneckerDelta(n,k) + c_e_c[n]*KroneckerDelta(n,k-N-1)
suma3 = Sum( p[n]*a*a_c.subs(k,j), (n, 0, N ) )

final = -i*summed + suma3.doit() # right hand side
LHS = rho_d[k,j]

rho_kk = final.subs(j,k) # derivative of rho_kk with respect to time

rho_kN1_k = final.subs( [(k,q+N+1) , (j,q)])

#refine(final, k >= 0 & k <= 2*N+1 & j>=0 & j<= 2*N + 1)



#M = M0[r,s]
#Res = Sum(H*M, (r, 0, n) ).doit()
#print( Res.subs(p,0) )