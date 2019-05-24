from galgebra import ga

import sympy

# Symbolic Variables
x = sympy.symbols(f'x1:{4}')
Dx = sympy.symbols(f'Dx1:{4}')

# Remark about variables
print(f'\nNOTE: Linear polynomials must be written in terms of the variables {x}.\n')

# 'Poisson' Matrices
P = sympy.MatrixSymbol('P',3,3)
P = sympy.Matrix(P)
Q = sympy.MatrixSymbol('Q',3,3)
Q = sympy.Matrix(Q)

# Bivector field coefficient
print(f'Enter the coefficients of the linear Poisson tensor P:\n')
for i in range(0, 2):
    for j in range(i + 1, 3):
        P[i,j] = str(input(f'P_{i + 1}{j + 1} = '))
print(f'\nEnter the coefficients of the linear Poisson tensor Q:\n')
for i in range(0, 2):
    for j in range(i + 1, 3):
        Q[i,j] = str(input(f'Q_{i + 1}{j + 1} = '))

# Trivial cases
if P[0,1] == 0 and P[0,2] == 0 and P[1,2] == 0 and Q[0,1] == 0 and Q[0,2] == 0 and Q[1,2] == 0:
    print(f'\nP and Q are trivial Poisson tensors. In particular, they are isomorphic.')
    quit()
if P[0,1] == 0 and P[0,2] == 0 and P[1,2] == 0:
    if Q[0,1] != 0 or Q[0,2] != 0 or Q[1,2] != 0:
        print(f'\nP = 0 and Q != 0. In particular, they are not isomorphic.')
        quit()
if Q[0,1] == 0 and Q[0,2] == 0 and Q[1,2] == 0:
    if P[0,1] != 0 or P[0,2] != 0 or P[1,2] != 0:
        print(f'\nP != 0 and Q = 0. In particular, they are not isomorphic.')
        quit()

# Matrix of P and Q
for i in range(0, 3):
    for j in range(i, 3):
        if i == j:
            P[i,j] = 0
        else:
            P[j,i] = (-1) * P[i,j]
for i in range(0, 3):
    for j in range(i, 3):
        if i == j:
            Q[i,j] = 0
        else:
            Q[j,i] = (-1) * Q[i,j]

# Jacobi identities
Jacobiator_P = sympy.simplify(sum(P[0, l] * sympy.diff(P[1, 2], x[l]) + P[1, l] * sympy.diff(P[2, 0], x[l]) + P[2, l] * sympy.diff(P[0, 1], x[l]) for l in range(3)))
if Jacobiator_P != 0:
    print(f'\nP is not a Poisson tensor:\n\n [P,P]_123 = {Jacobiator_P}')
Jacobiator_Q = sympy.simplify(sum(Q[0, l] * sympy.diff(Q[1, 2], x[l]) + Q[1, l] * sympy.diff(Q[2, 0], x[l]) + Q[2, l] * sympy.diff(Q[0, 1], x[l]) for l in range(3)))
if Jacobiator_Q != 0:
    print(f'\nQ is not a Poisson tensor:\n\n [Q,Q]_123 = {Jacobiator_Q}')

if Jacobiator_P != 0 or Jacobiator_Q != 0:
    quit()

pairing_E_P = sympy.expand(x[0] * P[1,2] + (-1) * x[1] * P[0,2] + x[2] * P[0,1])
hess_pairing_E_P = sympy.simplify(sympy.derive_by_array(sympy.derive_by_array(pairing_E_P, x),x)).tomatrix()
(Trnsf_1,diag_hess_P) = hess_pairing_E_P.diagonalize()
modular_vector_field_P = sympy.simplify(sum(sympy.simplify(sum(sympy.diff(P[i,j], x[i]) for i in range(3))) * Dx[j] for j in range(3)))

pairing_E_Q = sympy.expand(x[0] * Q[1,2] + (-1) * x[1] * Q[0,2] + x[2] * Q[0,1])
hess_pairing_E_Q = sympy.simplify(sympy.derive_by_array(sympy.derive_by_array(pairing_E_Q, x),x)).tomatrix()
(Trnsf_2,diag_hess_Q) = hess_pairing_E_Q.diagonalize()
modular_vector_field_Q = sympy.simplify(sum(sympy.simplify(sum(sympy.diff(Q[i,j], x[i]) for i in range(3))) * Dx[j] for j in range(3)))
#print(f'<E,P> = {pairing_E_P}')
#print(f'Hess<E,P> = {hess_pairing_E_P}')
#print(diag_hess)
#print(f'rank_hess = {hess_pairing_E_P.rank()}')
#print(f'Z = {modular_vector_field}')

# Classification P
classification_P = 0
if modular_vector_field_P == 0:
    if hess_pairing_E_P.rank() == 1:
        #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3')
        classification_P = 11
    if hess_pairing_E_P.rank() == 2:
        # Case index 2
        if diag_hess_P[0,0] * diag_hess_P[1,1] > 0 or diag_hess_P[0,0] * diag_hess_P[2,2] > 0 or diag_hess_P[1,1] * diag_hess_P[2,2] > 0:
            #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1')
            classification_P = 12
        else:
            # Case index 1
            #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 - x2*Dx3^Dx1')
            classification_P = 13
    if hess_pairing_E_P.rank() == 3:
        # Distinguish indexes of the Hessian matrix of P
        index_hess_sum_P = diag_hess_P[0,0]/abs(diag_hess_P[0,0]) + diag_hess_P[1,1]/abs(diag_hess_P[1,1])+diag_hess_P[2,2]/abs(diag_hess_P[2,2])
        if index_hess_sum_P == 3 or index_hess_sum_P == -3:
            #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1 + x3*Dx1^Dx2')
            classification_P = 14
        else:
            #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1 - x3*Dx1^Dx2')
            classification_P = 15
else:
    if hess_pairing_E_P.rank() == 0:
        #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = x2*Dx2^Dx3 - x1*Dx3^Dx1')
        classification_P = 16
    if hess_pairing_E_P.rank() == 2:
        # Case index 2
        if diag_hess_P[0,0] * diag_hess_P[1,1] > 0 or diag_hess_P[0,0] * diag_hess_P[2,2] > 0 or diag_hess_P[1,1] * diag_hess_P[2,2] > 0:
            #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = (4*a*x1 + x2)*Dx2^Dx3 + (4*a*x2 - x1)*Dx3^Dx1,\n\nwhere a > 0.')
            classification_P = 17
        else:
            # Case index 1
            #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = (4*a*x1 + x2)*Dx2^Dx3 - (x1 + 4*a*x2)*Dx3^Dx1,\n\nwhere a > 0.')
            classification_P = 18
    if hess_pairing_E_P.rank() == 1:
        #print(f'\nP is isomorphic via a linear isomorphism to:\n\n L = (4*x1 + x2)*Dx2^Dx3 - x1*Dx3^Dx1')
        classification_P = 19

# Classification Q
classification_Q = 0
if modular_vector_field_Q == 0:
    if hess_pairing_E_Q.rank() == 1:
        #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3')
        classification_Q = 11
    if hess_pairing_E_Q.rank() == 2:
        # Case index 2
        if diag_hess_Q[0,0] * diag_hess_Q[1,1] > 0 or diag_hess_Q[0,0] * diag_hess_Q[2,2] > 0 or diag_hess_Q[1,1] * diag_hess_Q[2,2] > 0:
            #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1')
            classification_Q = 12
        else:
            # Case index 1
            #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 - x2*Dx3^Dx1')
            classification_Q = 13
    if hess_pairing_E_Q.rank() == 3:
        # Distinguish indexes of the Hessian matrix of Q
        index_hess_sum_Q = diag_hess_Q[0,0]/abs(diag_hess_Q[0,0]) + diag_hess_Q[1,1]/abs(diag_hess_Q[1,1])+diag_hess_Q[2,2]/abs(diag_hess_Q[2,2])
        if index_hess_sum_Q == 3 or index_hess_sum_Q == -3:
            #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1 + x3*Dx1^Dx2')
            classification_Q = 14
        else:
            #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = x1*Dx2^Dx3 + x2*Dx3^Dx1 - x3*Dx1^Dx2')
            classification_Q = 15
else:
    if hess_pairing_E_Q.rank() == 0:
        #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = x2*Dx2^Dx3 - x1*Dx3^Dx1')
        classification_Q = 16
    if hess_pairing_E_Q.rank() == 2:
        # Case index 2
        if diag_hess_Q[0,0] * diag_hess_Q[1,1] > 0 or diag_hess_Q[0,0] * diag_hess_Q[2,2] > 0 or diag_hess_Q[1,1] * diag_hess_Q[2,2] > 0:
            #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = (4*a*x1 + x2)*Dx2^Dx3 + (4*a*x2 - x1)*Dx3^Dx1,\n\nwhere a > 0.')
            classification_Q = 17
        else:
            # Case index 1
            #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = (4*a*x1 + x2)*Dx2^Dx3 - (x1 + 4*a*x2)*Dx3^Dx1,\n\nwhere a > 0.')
            classification_Q = 18
    if hess_pairing_E_Q.rank() == 1:
        #print(f'\nQ is isomorphic via a linear isomorphism to:\n\n L = (4*x1 + x2)*Dx2^Dx3 - x1*Dx3^Dx1')
        classification_Q = 19

if classification_P == classification_Q:
    print(f'\nP and Q are isomorphic linear Poisson tensors.')
else:
    print(f'\nP and Q are not isomorphic linear Poisson tensors.')


