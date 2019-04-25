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
    print(f'\nNOTE: Functions must be written in terms of the variables {x}.\n')
else:
    print(f'\nNOTE: Functions must be written in terms of the variables ({x[0]},...,{x[m-1]}).\n')

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
    print(f'\nThe bivector field P is not a Poisson tensor.\n\nNOTE: [,] denote the Schouten-Nijenhuis bracket for multivector fields.')
    quit()

# '2-Form B' Matrix
B = sympy.MatrixSymbol('B',m,m)
B = sympy.Matrix(B)

# Closed 2-form B
print(f'3. Enter the coefficients of the closed 2-form B:\n')
for i in range(0, m - 1):
    for j in range(i + 1, m):
        B[i,j] = str(input(f'B_{i + 1}{j + 1} = '))

# Matrix of B
for i in range(0, m - 1):
    for j in range(i, m):
        if i == j:
            B[i,j] = 0
        else:
            B[j,i] = (-1) * B[i,j]
B[m - 1,m - 1] = 0

if B == sympy.zeros(m):
    print(f'\nIs B = 0. Therefore, gauge(P) = P.')
print(f'')
# Closedness of B
closedness_B_counter = 0
for i in range(0, m - 2):
    for j in range(i + 1, m - 1):
        for k in range(j + 1,m):
            closedness_B = sympy.simplify(sympy.diff(B[i, j], x[k]) + sympy.diff(B[j, k], x[i]) + sympy.diff(B[k, i], x[j]))
            if closedness_B != 0:
                print(f'dB_{i + 1}{j + 1}{k + 1} = {closedness_B}')
                closedness_B_counter = 1

if closedness_B_counter == 1:
    print(f'\nThe differential 2-form B is not closed.\n\nNOTE: Here, d denotes the exterior derivative for differential forms.')
    quit()

I_minus_BP = sympy.Matrix(sympy.simplify(sympy.eye(m)-B*P))
det_I_minus_BP = sympy.factor(sympy.simplify(I_minus_BP.det()))

if det_I_minus_BP == 0:
    print(f' det(I - B*P) == 0')
    print(f'\nThe differential 2-form B can\'t induces a gauge transformation. ')
    quit()
else:
    gauge_P_matrix = sympy.Matrix(sympy.simplify(P*(I_minus_BP.inv())))
    gauge_P = str(f'')
    for i in range(m-1):
        for j in range(i+1,m):
            gauge_P_aux = str(f'{gauge_P_matrix[i,j]} * {Dx[i]}^{Dx[j]}')
            gauge_P = gauge_P + str(f' + ') +gauge_P_aux
    print(f' gauge(P) = {gauge_P}')
    print(f'\nDefined in the domain {{{det_I_minus_BP} != 0}}.')

#print(det_I_minus_BP)
#print(gauge_P_matrix)
