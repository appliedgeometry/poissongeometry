from galgebra import ga

import sympy

# Symbolic Variables
x = sympy.symbols(f'x1:{4}')
Dx = sympy.symbols(f'Dx1:{4}')

# Remark about variables
print(f'\nNOTE: Linear polynomials must be written in terms of the variables {x}.\n')

# 'Poisson' Matrix
P = sympy.MatrixSymbol('P',3,3)
P = sympy.Matrix(P)

# Bivector field coefficient
print(f'1. Enter the coefficients of the linear Poisson tensor P:\n')
for i in range(0, 2):
    for j in range(i + 1, 3):
        P[i,j] = str(input(f'P_{i + 1}{j + 1} = '))

# Trivial case
if P[0,1] == 0 and P[0,2] == 0 and P[1,2] == 0:
    print(f'\nP is the trivial Poisson tensor.')
    quit()

# Matrix of P
for i in range(0, 2):
    for j in range(i, 3):
        if i == j:
            P[i,j] = 0
        else:
            P[j,i] = (-1) * P[i,j]
P[2,2] = 0

# Jacobi identity and definition of parameters
Jacobiator = sympy.simplify(sum(P[0, l] * sympy.diff(P[1, 2], x[l]) + P[1, l] * sympy.diff(P[2, 0], x[l]) + P[2, l] * sympy.diff(P[0, 1], x[l]) for l in range(3)))
if Jacobiator != 0:
    print(f'\nP is not a Poisson tensor:\n\n [P,P] = {Jacobiator}')
    quit()
else:
    pairing_E_P = sympy.expand(x[0] * P[1,2] + (-1) * x[1] * P[0,2] + x[2] * P[0,1])
    hess_pairing_E_P = sympy.simplify(sympy.derive_by_array(sympy.derive_by_array(pairing_E_P, x),x)).tomatrix()
    (Trnsf,diag_hess) = hess_pairing_E_P.diagonalize()
    modular_vector_field = sympy.simplify(sum(sympy.simplify(sum(sympy.diff(P[i,j], x[i]) for i in range(3))) * Dx[j] for j in range(3)))

#print(f'<E,P> = {pairing_E_P}')
#print(f'Hess<E,P> = {hess_pairing_E_P}')
#print(diag_hess)
#print(f'rank_hess = {hess_pairing_E_P.rank()}')
#print(f'Z = {modular_vector_field}')

# Classification
if modular_vector_field == 0:
    if hess_pairing_E_P.rank() == 1:
        print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3')
    if hess_pairing_E_P.rank() == 2:
        # Case index 2
        if diag_hess[0,0] * diag_hess[1,1] > 0 or diag_hess[0,0] * diag_hess[2,2] > 0 or diag_hess[1,1] * diag_hess[2,2] > 0:
            print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1')
        else:
            # Case index 1
            print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 - x2*Dx3^Dx1')
    if hess_pairing_E_P.rank() == 3:
        # Distinguish indexes of the Hessian matrix of P
        index_hess_sum = diag_hess[0,0]/abs(diag_hess[0,0]) + diag_hess[1,1]/abs(diag_hess[1,1])+diag_hess[2,2]/abs(diag_hess[2,2])
        if index_hess_sum == 3 or index_hess_sum == -3:
            print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1 + x3*Dx1^Dx2')
        else:
            print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1 - x3*Dx1^Dx2')
else:
    if hess_pairing_E_P.rank() == 0:
        print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x2*Dx2^Dx3 - x1*Dx3^Dx1')
    if hess_pairing_E_P.rank() == 2:
        # Case index 2
        if diag_hess[0,0] * diag_hess[1,1] > 0 or diag_hess[0,0] * diag_hess[2,2] > 0 or diag_hess[1,1] * diag_hess[2,2] > 0:
            print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = (4*a*x1 + x2)*Dx2^Dx3 + (4*a*x2 - x1)*Dx3^Dx1,\n\nwhere a > 0.')
        else:
            # Case index 1
            print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = (4*a*x1 + x2)*Dx2^Dx3 - (x1 + 4*a*x2)*Dx3^Dx1,\n\nwhere a > 0.')
    if hess_pairing_E_P.rank() == 1:
        print(
            f'\nP is isomorphic via a linear isomorphism to:\n\n L = (4*x1 + x2)*Dx2^Dx3 - x1*Dx3^Dx1')

