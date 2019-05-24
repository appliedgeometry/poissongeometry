from galgebra import ga

import sympy
import itertools

# Dimension
m = int(input(f'\nEnter the dimension:\n m = '))

# Symbolic Variables
x = sympy.symbols(f'x1:{m + 1}')
dx = sympy.symbols(f'dx1:{m + 1}')

# Remark about variables
if m < 4:
    print(f'\n NOTE: All functions must be written in terms of the variables {x}.\n')
else:
    print(f'\n NOTE: All functions must be written in terms of the variables ({x[0]},...,{x[m-1]}).\n')

# 'Poisson' Matrix
P = sympy.MatrixSymbol('P',m,m)
P = sympy.Matrix(P)

Q = sympy.MatrixSymbol('Q',m,m)
Q = sympy.Matrix(Q)

# Bivector field coefficient
print(f'Enter the coefficients of the Poisson tensor P:\n')
for i in range(0, m - 1):
    for j in range(i + 1, m):
        P[i,j] = str(input(f'P_{i + 1}{j + 1} = '))

print(f'\nEnter the coefficients of the Poisson tensor Q:\n')
for i in range(0, m - 1):
    for j in range(i + 1, m):
        Q[i,j] = str(input(f'Q_{i + 1}{j + 1} = '))

# Matrix of P adn Q
for i in range(0, m):
    for j in range(i, m):
        if i == j:
            P[i,j] = 0
        else:
            P[j,i] = (-1) * P[i,j]
for i in range(0, m):
    for j in range(i, m):
        if i == j:
            Q[i,j] = 0
        else:
            Q[j,i] = (-1) * Q[i,j]
print(f'')

# Jacobi identity
Jacobiator_counter_P = 0
Jacobiator_counter_Q = 0
for i in range(0, m - 2):
    for j in range(i + 1, m - 1):
        for k in range(j + 1,m):
            Jacobiator_P = sympy.simplify(sum(P[i, l] * sympy.diff(P[j, k], x[l]) + P[j, l] * sympy.diff(P[k, i], x[l]) + P[k, l] * sympy.diff(P[i, j], x[l]) for l in range(m)))
            if Jacobiator_P != 0:
                print(f'[P,P]_{i + 1}{j + 1}{k + 1} = {Jacobiator_P}')
                Jacobiator_counter_P = 1
if Jacobiator_counter_P == 1:
    print(f'\nThe bivector field P is not a Poisson tensor.\n')
for i in range(0, m - 2):
    for j in range(i + 1, m - 1):
        for k in range(j + 1,m):
            Jacobiator_Q = sympy.simplify(sum(Q[i, l] * sympy.diff(Q[j, k], x[l]) + Q[j, l] * sympy.diff(Q[k, i], x[l]) + Q[k, l] * sympy.diff(Q[i, j], x[l]) for l in range(m)))
            if Jacobiator_Q != 0:
                print(f'[Q,Q]_{i + 1}{j + 1}{k + 1} = {Jacobiator_Q}')
                Jacobiator_counter_Q = 1
if Jacobiator_counter_Q == 1:
    print(f'\nThe bivector field Q is not a Poisson tensor.\n')

# Poisson pair test
P_mas_Q = sympy.Matrix(sympy.simplify(P+Q))
Jacobiator_counter_P_mas_Q = 0
for i in range(0, m - 2):
    for j in range(i + 1, m - 1):
        for k in range(j + 1,m):
            Jacobiator_P_mas_Q = sympy.simplify(sum(P_mas_Q[i, l] * sympy.diff(P_mas_Q[j, k], x[l]) + P_mas_Q[j, l] * sympy.diff(P_mas_Q[k, i], x[l]) + P_mas_Q[k, l] * sympy.diff(P_mas_Q[i, j], x[l]) for l in range(m)))
            if Jacobiator_P_mas_Q != 0:
                print(f'[P+Q,P+Q]_{i + 1}{j + 1}{k + 1} = {Jacobiator_P_mas_Q}')
                Jacobiator_counter_P_mas_Q = 1
if Jacobiator_counter_P_mas_Q == 1:
    print(f'\nThe bivector field P is not a Poisson tensor.\n\n NOTE: [,] denote the Schouten-Nijenhuis bracket for multivector fields.')
else:
    print(f' [P+Q,P+Q] = 0\n\nThe bivector field P+Q is a Poisson tensor.\n\n NOTE: [,] denote the Schouten-Nijenhuis bracket for multivector fields.')


