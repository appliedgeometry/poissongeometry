from galgebra import ga

import sympy
import itertools

# Dimension
m = int(input(f'\n1. Enter the dimension:\n m = '))

# Symbolic Variables
x = sympy.symbols(f'x1:{m + 1}')
Dx = sympy.symbols(f'Dx1:{m + 1}')

# Remark about variables
if m < 4:
    print(f'\nNOTE: Homogeneous polynomials must be written in terms of the variables {x}.\n')
else:
    print(f'\nNOTE: Homogeneous polynomials must be written in terms of the variables ({x[0]},...,{x[m-1]}).\n')

# 'Poisson' Matrix
P = sympy.MatrixSymbol('P',m,m)
P = sympy.Matrix(P)

# Bivector field coefficient
print(f'2. Enter the coefficients of the Poisson tensor P:\n')
for i in range(0, m - 1):
    for j in range(i + 1, m):
        P[i,j] = str(input(f'P_{i + 1}{j + 1} = '))

# Matrix of P
for i in range(0, m - 1):
    for j in range(i, m):
        if i == j:
            P[i,j] = 0
        else:
            P[j,i] = (-1) * P[i,j]
P[m - 1,m - 1] = 0
print(f'')
# Jacobi identity
Jacobiator_counter = 0
for i in range(0, m - 2):
    for j in range(i + 1, m - 1):
        for k in range(j + 1,m):
            Jacobiator = sympy.simplify(sum(P[i, l] * sympy.diff(P[j, k], x[l]) + P[j, l] * sympy.diff(P[k, i], x[l]) + P[k, l] * sympy.diff(P[i, j], x[l]) for l in range(m)))
            if Jacobiator != 0:
                print(f'[P,P]_{i + 1}{j + 1}{k + 1} = {Jacobiator}')
                Jacobiator_counter = 1

if Jacobiator_counter == 1:
    print(f'\nThe bivector field P is not a Poisson tensor.\n\nNOTE: [,] denote the Schouten-Nijenhuis bracket.')
    quit()

# Modular vector field
print(f'3. Modular vector field of P relative to the Euclidean volume form on R^{m}:')
Z = sympy.simplify(sum(sympy.simplify(sum(sympy.diff(P[i,j], x[i]) for i in range(m))) * Dx[j] for j in range(m)))
print(f'\n Z = {Z}')
if Z == 0:
    print(f'\nThe Poisson tensor P is unimodular on R^{m}.')
else:
    print(f'\nThe Poisson tensor P is not unimodular on R^{m}.')