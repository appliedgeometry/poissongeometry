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
    print(f'\nNOTE: All functions must be written in terms of the variables {x}.\n')
else:
    print(f'\nNOTE: All functions must be written in terms of the variables ({x[0]},...,{x[m-1]}).\n')

# 'Poisson' Matrix
P = sympy.MatrixSymbol('P',m,m)
P = sympy.Matrix(P)

# Bivector field coefficient
print(f'Enter the coefficients of the Poisson tensor P:\n')
for i in range(0, m - 1):
    for j in range(i + 1, m):
        P[i,j] = str(input(f'P_{i + 1}{j + 1} = '))

# Matrix of P
for i in range(0, m):
    for j in range(i, m):
        if i == j:
            P[i,j] = 0
        else:
            P[j,i] = (-1) * P[i,j]
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
    print(f'\nThe bivector field P is not a Poisson tensor.\n\nNOTE: [,] denote the Schouten-Nijenhuis bracket for multivector fields.')
    quit()

# 1-Forms coefficient
print(f'Enter the coefficients of the 1-form alpha:\n')
r = 0
alpha = []
beta = []
while r < m:
    coeff_alpha = str(input(f'alpha_{r+1} = '))
    alpha.append(sympy.sympify(coeff_alpha))
    r = r + 1
print(f'\nEnter the coefficients of the 1-form beta:\n')
r = 0
while r < m:
    coeff_beta = str(input(f'beta_{r+1} = '))
    beta.append(sympy.sympify(coeff_beta))
    r = r + 1

bracket_alpha_beta = []
for k in range(m):
    bracket_alpha_P_D_beta = sympy.simplify(sum(sum(alpha[i] * P[i,j] for i in range(m)) * sympy.diff(beta[k], x[j]) for j in range(m)))
    bracket_D_alpha_P_beta = sympy.simplify(sum(sympy.diff(alpha[k], x[i]) * sum(P[i, j] * beta[j] for j in range(m)) for i in range(m)))
    bracket_alpha_D_P_beta = sympy.simplify(sum(sum(alpha[i] * sympy.diff(P[i,j], x[k]) for i in range(m)) * beta[j] for j in range(m)))
    bracket_alpha_beta.append(sympy.simplify(bracket_alpha_P_D_beta + bracket_D_alpha_P_beta + bracket_alpha_D_P_beta))

print(f'\nThe bracket induced by P is\n\n {{alpha,beta}}_P = {sympy.simplify(sympy.Matrix(bracket_alpha_beta).dot(sympy.Matrix(dx)))}')


