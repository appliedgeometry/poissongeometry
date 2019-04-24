from galgebra import ga

import sympy
import itertools

# Dimension
m = int(input(f'\nEnter the dimension:\n m = '))

# Symbolic Variables
x = sympy.symbols(f'x1:{m + 1}')
Dx = sympy.symbols(f'Dx1:{m + 1}')

# Remark about variables
if m < 4:
    print(f'\nNOTE: All functions must be written in terms of the variables {x}.\n')
else:
    print(f'\nNOTE: All functions must be written in terms of the variables ({x[0]},...,{x[m-1]}).\n')

# 'Poisson' Matrix
P = sympy.MatrixSymbol('P',m,m)
P = sympy.Matrix(P)

# Bivector field coefficient
print(f'1. Enter the coefficients of the bivector field P:\n')
for i in range(0, m - 1):
    for j in range(i + 1, m):
        P[i,j] = str(input(f'P_{i + 1}{j + 1} = '))

# Bivector field P
print(f'\n2. Bivector Field P: Pendiente!')

# Matrix of P
print(f'\n3. Matrix of P:\n')
for i in range(0, m - 1):
    for j in range(i, m):
        if i == j:
            P[i,j] = 0
        else:
            P[j,i] = (-1) * P[i,j]
P[m - 1,m - 1] = 0

print(f'P = {P}')

# Jacobi identity
print(f'\n4. Schouten-Nijenhuis Bracket [P,P]:\n')
Jacobiator_counter = 0
for i in range(0, m - 2):
    for j in range(i + 1, m - 1):
        for k in range(j + 1,m):
            Jacobiator = sympy.simplify(sum(P[i, l] * sympy.diff(P[j, k], x[l]) + P[j, l] * sympy.diff(P[k, i], x[l]) + P[k, l] * sympy.diff(P[i, j], x[l]) for l in range(m)))
            if Jacobiator != 0:
                print(f'[P,P]_{i + 1}{j + 1}{k + 1} = {Jacobiator}')
                Jacobiator_counter = 1

if Jacobiator_counter == 1:
    print(f'\nThe bivector field P is not a Poisson tensor.')
else:
    print(f' [P,P] = 0\n\nThe bivector field P is a Poisson tensor.')

# Morphism P^# and kernel of P
print(f'\n5. Vector bundle morphism P^#:\n')
print(f'Enter the coefficients of the 1-forma alpha:\n')
r = 0
one_form_alpha = []
while r < m:
    coeff_one_form_alpha = str(input(f'alpha_{r+1} = '))
    one_form_alpha.append(sympy.sympify(coeff_one_form_alpha))
    r = r + 1
P_sharp_alpha = sympy.simplify(sum(sympy.simplify(sum(P[i, j] * one_form_alpha[i] for i in range(m))) * Dx[j] for j in range(m)))

print(f'\n P^#(alpha) = {P_sharp_alpha}')
if P_sharp_alpha == 0:
    print(f'\nIn particular, alpha \in Ker(P).')

# Poisson bracket
print(f'\n6. Poisson bracket {{f,g}}, induced by P, of two function f and g:\n')
f = str(input(f'f = '))
g = str(input(f'g = '))
bracket_f_g = 0
for i in range(0, m - 1):
    for j in range(i + 1, m):
        bracket_f_g_aux = sympy.simplify(P[i,j] * (sympy.diff(f, x[i]) * sympy.diff(g, x[j]) - sympy.diff(f, x[j]) * sympy.diff(g, x[i])))
        bracket_f_g = sympy.simplify(bracket_f_g + bracket_f_g_aux)

print(f'\n{{f,g}} = {bracket_f_g}')

# Hamiltonian vector field
print(f'\n7. Hamiltonian vector field X_h, relative to P, of a function h:\n')
h = str(input(f'h = '))
X_h = sum(sympy.simplify(sum(P[i, j] * sympy.diff(h, x[i]) for i in range(m))) * Dx[j] for j in range(m))

print(X_h)

# Casimir function
print(f'\n8. Determines if a function K is a Casimir function of P:\n')
K = str(input(f'K = '))
Casim_K = sum(sympy.simplify(sum(P[i, j] * sympy.diff(h, x[i]) for i in range(m))) * Dx[j] for j in range(m))
if Casim_K == 0:
    print(f'\nK is a Casimir function of P.')
else:
    print(f'\nK is not a Casimir function of P.')

# Poisson vector field
print(f'\n9. Determines if a vector field W is Poisson, that is, [W,P] = 0.')
print(f'Enter the coefficients of W:\n')
W = []
for i in range(0, m):
        W_aux = str(input(f'W_{i+1} = '))
        W.append(sympy.sympify(W_aux))
Schouten_W_P_counter = 0
print(f'')
for i in range(0, m - 1):
    for j in range(i + 1, m):
        Schouten_W_P = sympy.simplify(sum(W[k] * sympy.diff(P[i,j], x[k]) - P[i, k] * sympy.diff(W[j], x[k]) + P[j,k] * sympy.diff(W[i], x[k]) for k in range(m)))
        if Schouten_W_P != 0:
            print(f'[W,P]_{i + 1}{j + 1} = {Schouten_W_P}')
            Schouten_W_P_counter = 1

if Schouten_W_P_counter == 1:
    print(f'\nW is not a Poisson vector field.')
else:
    print(f'[W,P] = 0\n\nW is a Poisson field.')

# Modular vector field
print(f'\n10. Modular vector field Z of P respect the volumen form a*V0. Where V0 is the Euclidean volume form on R^{m} and a is a nonzero function.')
a = str(input(f'\nEnter the nonzero function:\n a = '))
X_ln_a = 1/(sympy.simplify(a)) * (-1) * sum(sympy.simplify(sum(P[i, j] * sympy.diff(a, x[j]) for j in range(m))) * Dx[i] for i in range(m))
Z_0 = sum(sympy.simplify(sum(sympy.diff(P[i,j], x[i]) for i in range(m))) * Dx[j] for j in range(m))
Z = sympy.simplify(Z_0 - X_ln_a)

print(f'\n Z_(a*V0) = {Z}')
if Z == 0:
    print(f'\nThe Poisson tensor P is unimodular on R^{m}.')

# Flascka-Ratiu

#a = 1
#K = []
#while a <= (m - 2):
#    Casimir_function = str(input(f'Enter the function K{a} = '))
#    K.append(sympy.sympify(Casimir_function))
#    a = a + 1

#matrix_grad_K = sympy.Matrix([])
#for i in range(0, m - 2):
#    grad_K = sympy.derive_by_array(K[i], x)
#    matrix_grad_K = sympy.Matrix([matrix_grad_K,grad_K])
    #matrix_grad_K = matrix_grad_K.row_insert(i, sympy.Matrix([grad_K]))
