"""
    Copyright 2019 by P Suarez-Serrato, Jose Ruíz and M Evangelista-Alvarado.
    Instituto de Matemáticas (UNAM-CU) México
    This is free software; you can redistribute it and/or
    modify it under the terms of the MIT License,
    https://en.wikipedia.org/wiki/MIT_License.

    This software has NO WARRANTY, not even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    For more information about this project, you can see the paper
    https://arxiv.org/abs/1912.01746

    Thanks for citing our work if you use it.
    @misc{evangelistaalvarado2019computational,
        title={On Computational Poisson Geometry I: Symbolic Foundations},
        author={M. A. Evangelista-Alvarado and J. C. Ruíz-Pantaleón and P. Suárez-Serrato},
        year={2019},
        eprint={1912.01746},
        archivePrefix={arXiv},
        primaryClass={math.DG}
    }
"""
from __future__ import unicode_literals

import sympy as sym
from sympy.matrices import zeros
import itertools as itools
from poisson.utils import (validate_dimension, symbolic_expression, show_coordinates,
                           two_tensor_form_to_matrix, remove_values_zero, basis
                           )


class PoissonGeometry:
    """ This class provides some useful tools for Poisson-Nijenhuis calculus on Poisson manifolds."""

    def __init__(self, dimension, variable='x'):
        # Obtains the dimension
        self.dim = validate_dimension(dimension)
        # Define what variables the class will work with
        self.variable = variable
        # Create the symbolics symbols
        self.coords = sym.symbols(f'{self.variable}1:{self.dim + 1}')
        # Show the coordinates with that will the class works
        self.coordinates = show_coordinates(self.coords)
        self.Dx_basis = basis(self.dim, self.coords, self.variable)

    def bivector_to_matrix(self, bivector, latex_format=False):
        """ Constructs the matrix of a 2-contravariant tensor field or bivector field.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is a symbolic skew-symmetric matrix of dimension (dim)x(dim) if latex_format is False
            in otherwise the result is the same but in latex format.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> pg.bivector_to_matrix(bivector, latex_format=False)
            >>> Matrix([[0, x3, -x2], [-x3, 0, x1], [x2, -x1, 0]])
            >>> pg.bivector_to_matrix(bivector, latex_format=True)
            >>> '\\left[\\begin{matrix}0 & x_{3} & - x_{2}\\\\- x_{3} & 0 & x_{1}\\\\x_{2} &
                - x_{1} & 0\\end{matrix}\\right]'
        """
        # Creates a symbolic Matrix
        bivector_matrix = zeros(self.dim)
        # Assigns the corresponding coefficients of the bivector field
        for index in bivector.keys():
            i, j = index
            # Makes the Poisson Matrix
            bivector_matrix[i-1, j-1] = sym.sympify(bivector[(i, j)])
            bivector_matrix[j-1, i-1] = (-1) * bivector_matrix[i-1, j-1]

        # Return a symbolic Poisson matrix or the same expression in latex format
        if latex_format:
            return sym.latex(bivector_matrix)
        return bivector_matrix

    def sharp_morphism(self, bivector, one_form, latex_format=False, remove_zeros=True):
        """ Calculates the image of a differential 1-form under the vector bundle morphism 'sharp' P#: T*M -> TM
            defined by P#(alpha) := i_(alpha)P, where P is a Poisson bivector field on a manifold M, alpha a 1-form
            on M and i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :one_form:
            Is a 1-form differential in a dictionary format with tuple type 'keys' and string type 'values'.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is the image of a differential 1-form alpha under the vector bundle morphism 'sharp' in a
            dictionary format with tuple type 'keys' and symbol type 'values'

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # For one form a1*x1*dx1 + a2*x2*dx2 + a3*x3*dx3.
            >>> one_form = {(1,): 'a1*x1', (2,): 'a2*x2', (3,): 'a3*x3'}
            >>> # P#(one_form) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
            >>> sharp_morphism(bivector, one_form, latex_format=False)
            >>> {(1,): x2*(-a2*x3 + a3*x3), (2,): x1*(a1*x3 - a3*x3), (3,): x1*x2*(-a1 + a2)}
            >>> sharp_morphism(bivector, one_form, latex_format=True)
            >>> 'x_{2} x_{3} \\left(- a_{2} + a_{3}\\right) \\boldsymbol{Dx}_{1} + x_{1} x_{3} \\left
                (a_{1} - a_{3}\\right) \\boldsymbol{Dx}_{2} + x_{1} x_{2} \\left(- a_{1} + a_{2}\right)
                \\boldsymbol{Dx}_{3}'
        """
        # Converts strings to symbolic variables
        one_form = {i: sym.sympify(coeff_i) for i, coeff_i in one_form.items()}
        bivector = {ij: sym.sympify(coeff_ij) for ij, coeff_ij in bivector.items()}

        """
            Calculation of the vector field bivector^#(one_form) as
                ∑_{i}(∑_{j} bivector_ij (one_form_i*Dx_j - one_form_j*Dx_i)
            where i < j.
        """
        p_sharp = {}
        for ij, bivector_ij in bivector.items():
            # Get the values i and j from bivector index ij
            i, j = ij
            p_sharp_aux_i = p_sharp.get((i,), 0)
            p_sharp_aux_j = p_sharp.get((j,), 0)
            one_form_i = one_form.get((i,), 0)
            one_form_j = one_form.get((j,), 0)
            # Calculates one form i*Pij*Dxj
            p_sharp[(j,)] = sym.simplify(p_sharp_aux_j + one_form_i * bivector_ij)
            # Calculates one form j*Pji*Dxi
            p_sharp[(i,)] = sym.simplify(p_sharp_aux_i - one_form_j * bivector_ij)

        # Return a vector field expression in LaTeX format
        if latex_format:
            # For copy and paste this result in a latex editor, do not forget make "print()" in the result.
            return symbolic_expression(p_sharp, self.dim, self.coords, self.variable).Mv_latex_str()
        # Return a symbolic dictionary.
        return remove_values_zero(p_sharp) if remove_zeros else p_sharp

    def is_in_kernel(self, bivector, one_form):
        """ Check if a differential 1-form alpha belongs to the kernel of a given Poisson bivector field,
            that is check if P#(alpha) = 0

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :one_form:
            Is a 1-form differential in a dictionary format with tuple type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if P#(alpha) = 0, in other case is False.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For one form x1*dx1 + x2*dx2 + x3*dx3.
            >>> one_form = {(1): 'x1', (2): 'x2', (3): 'x3'}
            >>> # P#(one_form) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
            >>> pg.is_in_kernel(bivector, one_form)
            >>> True
        """
        p_sharp = self.sharp_morphism(bivector, one_form)
        # Converts a dictionary symbolic to a symbolic expression and verify is zero with a sympy method
        return True if symbolic_expression(p_sharp, self.dim, self.coords, self.variable).is_zero() else False

    def hamiltonian_vf(self, bivector, hamiltonian_function, latex_format=False, remove_zeros=True):
        """ Calculates the Hamiltonian vector field of a function relative to a Poisson bivector field as follows:
            X_h = P#(dh), where d is the exterior derivative of h and P#: T*M -> TM is the vector bundle morphism
            defined by P#(alpha) := i_(alpha)P, with i is the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :hamiltonian_function:
            Is a function scalar h: M --> R that is a string type.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is the hamiltonian vector field relative to a Poisson in a dictionary format with tuple type
            'keys' and symbol type 'values'

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For hamiltonian_function h(x1,x2,x3) = x1 + x2 + x3.
            >>> hamiltonian_function = 'x1 + x2 + x3'
            >>> # X_h = P#(dh) = (x2 - x3)*Dx1 + (-x1 + x3)*Dx2 + (x1 - x2)*Dx3.
            >>> pg.hamiltonian_vf(bivector, hamiltonian_function, latex_format=False)
            >>> {(1,): x2 - x3, (2,): -x1 + x3, (3,): x1 - x2}
            >>> pg.hamiltonian_vf(bivector, hamiltonian_function, latex_format=True)
            >>> \\left ( x_{2} - x_{3}\\right ) \\boldsymbol{Dx}_{1} + \\left ( - x_{1} + x_{3}\\right )
                \\boldsymbol{Dx}_{2} + \\left ( x_{1} - x_{2}\\right ) \\boldsymbol{Dx}_{3}'
        """
        # Converts a string to symbolic expression
        h = sym.sympify(hamiltonian_function)
        # Calculates the differential of hamiltonian_function
        dh = sym.derive_by_array(h, self.coords)
        # Calculates the Hamiltonian vector field
        keys_list = [(index,) for index in range(1, self.dim + 1)]
        hamiltonian_vf = self.sharp_morphism(bivector, dict(zip(keys_list, dh)))

        if latex_format:
            # return to symbolic expression in LaTeX format
            return symbolic_expression(hamiltonian_vf, self.dim,
                                       self.coords, self.variable).Mv_latex_str()
        else:
            # return a symbolic dictionary
            return remove_values_zero(hamiltonian_vf) if remove_zeros else hamiltonian_vf

    def is_casimir(self, bivector, function):
        """ Check if a function is a Casimir function of a given (Poisson) bivector field, that is
            P#(dK) = 0, where dK is the exterior derivative (differential) of K and P#: T*M -> TM is the
            vector bundle morphism defined by P#(alpha) := i_(alpha)P, with i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :function:
            Is a function scalar f: M --> R that is a string type.

        Returns
        =======
            The result is True if P#(df) = 0, in other case is False.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For function f(x1,x2,x3) = x1^2 + x2^2 + x3^2.
            >>> function = 'x1**2 + x2**2 + x3**2'
            >>> pg.is_casimir(bivector, function)
            >>> True
        """

        # Calculates the Hamiltonian vector field
        hamiltonian_vf = self.hamiltonian_vf(bivector, function, remove_zeros=False)
        # Converts a dictionary symbolic to a symbolic expression and verify is zero with a method of sympy
        return True if symbolic_expression(hamiltonian_vf, self.dim,
                                           self.coords, self.variable).is_zero() else False

    def poisson_bracket(self, bivector, f, g, latex_format=False):
        """ Calculates the poisson bracket {f,g} = π(df,dg) = ⟨dg,π#(df)⟩ of two functions
            f and g on a Poisson manifold (M,P), where d is the exterior derivatives and P#: T*M -> TM is the
            vector bundle morphism defined by P#(alpha) := i_(alpha)P, with i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :f / g:
            Is a function scalar f/g: M --> R that is a string type.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is the poisson bracket from f with g in a dictionary format with tuple type 'keys' and
            symbol type 'values' if latex format is False or in otherwise is the same result but in latex format

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For f(x1,x2,x3) = x1 + x2 + x3.
            >>> f = 'x1 + x2 + x3'
            >>> # For g(x1,x2,x3) = 'a1*x1 + a2*x2 + a3*x3'.
            >>> g = 'a1*x1 + a2*x2 + a3*x3'
            >>> # {f, g} = a1*(x2 - x3) - a2*(x1 - x3) + a3*(x1 - x2)
            >>> pg.poisson_bracket(bivector, f, g, latex_format=False)
            >>> a1*(x2 - x3) - a2*(x1 - x3) + a3*(x1 - x2)
            >>> pg.poisson_bracket(bivector, f, g, latex_format=True)
            >>> 'a_{1} \\left(x_{2} - x_{3}\\right) - a_{2} \\left(x_{1} - x_{3}\\right) + a_{3} \\left(x_{1}
                - x_{2}\\right)'
        """
        # Convert from string to sympy value
        f, g = sym.sympify(f), sym.sympify(g)

        if f == g:
            return 0  # {f,g} = 0 if f=g
        # Calculates the differentials of f and g
        keys_list = [(index,) for index in range(1, self.dim + 1)]
        df = dict(zip(keys_list, sym.derive_by_array(f, self.coords)))
        dg = dict(zip(keys_list, sym.derive_by_array(g, self.coords)))
        # Calculates {f,g} = <g,P#(df)> = ∑_{i} (P#(df))^i * (dg)_i
        bracket_f_g = sym.simplify(sum(self.sharp_morphism(
                                                bivector,
                                                df,
                                                remove_zeros=False)[index] * dg[index] for index in df.keys()))

        # Return a symbolic type expression or the same expression in latex format
        return sym.latex(bracket_f_g) if latex_format else bracket_f_g

    def curl_operator(self, multivector, function, latex_format=False):
        """ Calculate the divergence of multivertoc field.

        Parameters
        ==========
        :multivector:
            Is a multivector filed in a dictionary format with integer type 'keys' and string type 'values'.
        :function:
            Is a nowhere vanishing function in a string type. If the function is constant you can input the
            number type.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            Returns the divergence of multivertor field in a dictionary format with integer type 'keys'
            and string type 'values' if latex format is False or in otherwise is the same result but in latex format

        Example
        ========
            >>> # Instance the class to dimension 6
            >>> pg = PoissonGeometry(6)
            >>> multivector = {(1,2): ‘x1*x2’, (1,3): ‘-x1*x3’, (2,3): ‘x2*x3’, (3,4): ‘1’, (3,5): ‘-1’, (4,6): ‘1’}
            >>> # If the function is constant only input the number.
            >>> function = 1
            >>> pg.curl_operator(multivector, function)
            >>> {0: 0}
            >>> pg.curl operator(multivector, function,latex_format=False)
            >>> '0'
        """

        if isinstance(multivector, str):
            return {0: 0}
        else:
            if isinstance(next(iter(multivector)), int):
                return sym.simplify(sum(sym.diff(multivector[i], self.coords[i - 1]) for i in range(1, self.dim + 1)))
            else:
                deg_multivector = len(next(iter(multivector)))
                curl_multivector = dict()
                for z in itools.combinations(range(1, self.dim + 1), deg_multivector - 1):
                    curl_multivector[z] = 0
                for key in multivector:
                    for j in range(1, deg_multivector + 1):
                        index = tuple(k for l, k in enumerate(key) if l not in [j - 1])
                        curl_multivector[index] = sym.simplify(curl_multivector[index] + (-1)**(j) * (sym.diff(sym.sympify(multivector[key]), self.coords[list(key).pop(j - 1) - 1]) + 1/(sym.sympify(function)) * sym.sympify(multivector[key]) * sym.diff(sym.sympify(function), self.coords[list(key).pop(j - 1) - 1])))  # noqa
                if not latex_format:
                    return remove_values_zero(curl_multivector)
                else:
                    return symbolic_expression(remove_values_zero(curl_multivector), self.dim,
                                               self.coords, self.variable).Mv_latex_str()

    def lichnerowicz_poisson_operator(self, bivector, multivector, latex_format=False):
        """ Calculates the Schouten-Nijenhuis bracket between a given (Poisson) bivector field and a (arbitrary)
            multivector field.
            The Lichnerowicz-Poisson operator is defined as
                [P,A](df1,...,df(a+1)) = sum_(i=1)^(a+1) (-1)**(i)*{fi,A(df1,...,î,...,df(a+1))}_P
                                     + sum(1<=i<j<=a+1) (-1)**(i+j)*A(d{fi,fj}_P,..î..^j..,df(a+1))
            where P = Pij*Dxi^Dxj (i < j), A = A^J*Dxj_1^Dxj_2^...^Dxj_a.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :multivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values' if
            dim(multivector) > 1 in other case is a string value.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is the Schouten-Nijenhuis bracket from bivector with multivector in a dictionary format with
            tuple type 'keys' and symbol type 'values' if latex format is False or in otherwise is the same result
            but in latex format

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For multivector (x1*x2*x3)*Dx1^Dx2^Dx3
            >>> multivector = {(1,2,3): 'x1*x2*x3'}
            >>> # [bivector, multivector] = 0
            >>> pg.lichnerowicz_poisson_operator(bivector, multivector, latex_format=False)
            >>> 0
            >>> pg.lichnerowicz_poisson_operator(self, bivector, multivector latex_format=True)
            >>> '0'
        """

        mltv = multivector
        # Degree of multivector
        if isinstance(list(multivector.keys())[0], int):
            deg_mltv = 1
        else:
            deg_mltv = len(list(multivector.keys())[0])

        if deg_mltv + 1 > self.dim:
            return 0
        else:
            # In this case, multivector is a function
            if isinstance(multivector, str):
                # [P,f] = -X_f, for any function f.
                return self.hamiltonian_vf(bivector, (-1) * sym.sympify(multivector))
            else:
                # A dictionary for the first term of [P,A], A a multivector
                schouten_biv_mltv_1 = {}
                lich_poiss_aux_1 = 0
                # The first term of [P,A] is a sum of Poisson brackets {xi,A^J}
                range_list = range(1, self.dim + 1)
                for z in itools.combinations(range_list, deg_mltv + 1):
                    for i in z:
                        # List of indices for A without the index i
                        # nw_idx = [e for e in z if e not in [i]]
                        nw_idx = [e for e in z if e != i]
                        # Calculates the Poisson bracket {xi,A^J}, J == nw_idx
                        mltv_nw_idx = mltv.get(tuple(nw_idx), 0)
                        lich_poiss_aux_11 = sym.simplify(
                            (-1)**(z.index(i)) * self.poisson_bracket(
                                bivector,
                                self.coords[i - 1],
                                sym.sympify(mltv_nw_idx))
                        )
                        lich_poiss_aux_1 = sym.simplify(lich_poiss_aux_1 + lich_poiss_aux_11)
                    # Add the brackets {xi,A^J} in the list schouten_biv_mltv_1
                    schouten_biv_mltv_1.update({z: lich_poiss_aux_1})
                    lich_poiss_aux_1 = 0
                # A dictionary for the second term of [P,A]
                schouten_biv_mltv_2 = {}
                sum_i = 0
                sum_y = 0
                # Scond term of [P,A] is a sum of Dxk(P^iris)*A^((J+k)\iris))
                for z in itools.combinations(range_list, deg_mltv + 1):
                    for y in itools.combinations(z, 2):
                        for i in range_list:
                            # List of indices for A without the indices in y
                            nw_idx_mltv = [e for e in z if e not in y]
                            # Add the index i
                            nw_idx_mltv.append(i)
                            # Sort the indices
                            nw_idx_mltv.sort()
                            # Ignore index repetition ADD SET FUNCTION
                            bivector_y = bivector.get(y, 0)
                            mltv_nw_idx_mltv = mltv.get(tuple(nw_idx_mltv), 0)
                            if nw_idx_mltv.count(i) != 2:
                                sum_i_aux = sym.simplify(
                                    (-1)**(z.index(y[0]) + z.index(y[1]) + nw_idx_mltv.index(i))
                                    * sym.diff(sym.sympify(bivector_y), self.coords[i - 1])
                                    * sym.sympify(mltv_nw_idx_mltv))
                                sum_i = sym.simplify(sum_i + sum_i_aux)
                        sum_y_aux = sum_i
                        sum_y = sym.simplify(sum_y + sum_y_aux)
                        sum_i = 0
                        sum_i_aux = 0
                    # Add the terms Dxk(P^iris)*A^((J+k)\iris)) in the list schouten_biv_mltv_2
                    schouten_biv_mltv_2.update({z: sum_y})
                    sum_y = 0
                    sum_y_aux = 0
                # A dictionary for the the Schouten bracket [P,A]
                schouten_biv_mltv = {}
                # Sum and add the terms in schouten_biv_mltv_1 and schouten_biv_mltv_2
                for ky in schouten_biv_mltv_1.keys():
                    schouten_biv_mltv.update({ky: sym.simplify(schouten_biv_mltv_1[ky] + schouten_biv_mltv_2[ky])})
                # return a dictionary or latex format.
                if latex_format:
                    return symbolic_expression(schouten_biv_mltv, self.dim, self.coords, self.variable).Mv_latex_str()
                else:
                    return remove_values_zero(schouten_biv_mltv)

    def jacobiator(self, bivector, latex_format=False):
        """ Calculates de Schouten-Nijenhuis bracket of a given bivector field with himself, that is [P,P]
            where [,] denote the Schouten bracket for multivector fields.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is the jacobiator of P with P in a dictionary format with tuple type 'keys' and
            symbol type 'values' if latex format is False or in otherwise is the same result but in latex format

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # Calculates [P, P]
            >>> pg.jacobiator(bivector, latex_format=False)
            >>> {0 : 0}
            >>> pg.jacobiator(bivector, latex_format=True)
            >>> '0'
        """
        return self.lichnerowicz_poisson_operator(bivector, bivector, latex_format=latex_format)

    def is_poisson_bivector(self, bivector):
        """ Calculates de Schouten-Nijenhuis bracket of a given bivector field with himself,
            that is calculates [P,P]_SN.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if [P,P] = 0, in other case is False.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # [P, P]
            >>> pg.is_poisson_bivector(bivector):
            >>> True
        """
        bivector_result = self.lichnerowicz_poisson_operator(bivector, bivector)
        return True if symbolic_expression(bivector_result, self.dim, self.coords, self.variable).is_zero() else False

    def is_poisson_vf(self, bivector, vector_field):
        """ Check if a vector field is a Poisson vector field of a given Poisson bivector field, that is calculate if
            [Z,P] = 0, where Z is vector_field variable and P is bivector variable.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :vector_field:
            Is a vector field in a dictionary format with tuple type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if [Z,P] = 0, in other case is False.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For vector field x1*Dx1 + x2*Dx2 + x3*Dx
            >>> vector_field = {(1,): 'x1', (2,): 'x2', (3,): 'x3'}
            >>> # [Z, P]
            >>> pg.is_poisson_vf(bivector, vector_field)
            >>> False
        """
        bivector_result = self.lichnerowicz_poisson_operator(bivector, vector_field)
        return True if symbolic_expression(bivector_result, self.dim, self.coords, self.variable).is_zero() else False

    def is_poisson_pair(self, bivector_1, bivector_2):
        """ Check if the sum of two Poisson bivector fields P1 and P2 is a Poisson bivector field, that is
            calculate if [P1,P2] = 0

        Parameters
        ==========
        :bivector_1 / bivector_2:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if [P1,P2] = 0, in other case is False.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector a*x1*x2*Dx1^Dx2 - b*x1*x3*Dx1^Dx3 + b*x1*x3*Dx2^Dx3
            >>> P = {(1,2): 'a*x1*x2', (1,3): '-b*x1*x3', (2,3): 'b*x2*x3'}
            >>> # For vector field x3**2*x1*Dx1
            >>> Q = {(1,2): 'x3**2'}
            >>> # [P, Q]
            >>> pg.is_poisson_pair(P, Q)
            >>> True
        """
        bivector_result = self.lichnerowicz_poisson_operator(bivector_1, bivector_2)
        return True if symbolic_expression(bivector_result, self.dim, self.coords, self.variable).is_zero() else False

    def modular_vf(self, bivector, function, latex_format=False):
        """ Calculates the modular vector field Z of a given Poisson bivector field P relative to the volume form
            f*Omega_0 defined as Z(g):= div(X_g) where Omega_0 = dx1^...^dx('dim'), f a non zero function and div(X_g)
            is the divergence respect to Omega of the Hamiltonian vector field
            X_g of f relative to P.
            Clearly, the modular vector field is Omega-dependent. If h*Omega is another volume form, with h a nowhere
            vanishing function on M, then
                Z' = Z - (1/h)*X_h,
            is the modular vector field of P relative to h*Omega.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :function:
            Is a function scalar h: M --> R that is a non zero and is a string type.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is modular vector field Z of a given Poisson bivector field P relative to the volume form
            in a dictionary format with tuple type 'keys' and symbol type 'values' if latex format is False or
            in otherwise is the same result but in latex format

        Example
        =======
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For the function exp:R^3 --> R
            >>> function = 'exp(x1 + x2 + x3)'
            >>> # Calculates Z' = Z - (1/h)*X_h that is (-x2 + x3)*Dx1 + (x1 - x3)*Dx2 + (-x1 + x2)*Dx3
            >>> pg.modular_vf(bivector, function)
            >>> {(1,): -x2 + x3, (2,): x1 - x3, (3,): -x1 + x2}
            >>> pg.modular_vf(bivector, function, latex_format=True)
            >>> '\\left ( - x_{2} + x_{3}\\right ) \\boldsymbol{Dx}_{1} + \\left ( x_{1} - x_{3}\\right ) \\boldsy
                mbol{Dx}_{2} + \\left ( - x_{1} + x_{2}\\right ) \\boldsymbol{Dx}_{3}'
        """
        # Converts strings to symbolic variables
        bivector = {key: (-1)*(sym.sympify(value)) for key, value in bivector.items()}
        modular_vector_field = self.curl_operator(bivector, function)
        if latex_format:
            return symbolic_expression(modular_vector_field, self.dim, self.coords, self.variable).Mv_latex_str()
        else:
            return modular_vector_field

    def is_homogeneous_unimodular(self, bivector):
        """ Check if a homogeneous Poisson bivector field is unimodular
            or not.

        Parameters
        ==========
        :bivector:
            Is a homogeneous Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if the homogeneous Poisson bivector fiel is unimodular, in other case is False.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> pg.is_homogeneous_unimodular(bivector)
            >>> True
        """
        modular_vf = self.modular_vf(bivector, 1)
        return True if symbolic_expression(modular_vf, self.dim, self.coords, self.variable).is_zero() else False

    def one_forms_bracket(self, bivector, one_form_1, one_form_2, latex_format=False):
        """ Calculates the Lie bracket of two differential 1-forms induced by a given Poisson bivector field.
            as {alpha,beta}_P := i_P#(alpha)(d_beta) - i_P#(beta)(d_alpha) + d_P(alpha,beta) for 1-forms alpha
            and beta.
            Where d_alpha and d_beta are the exterior derivative of alpha and beta, respectively, i_ the interior
            product of vector fields on differential forms, P#: T*M -> TM the vector bundle morphism defined by
            P#(alpha) := i_(alpha)P, with i the interior product of alpha and P. Note that, by definition
            {df,dg}_P = d_{f,g}_P, for ant functions f,g on M.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :one_form_1/one_form_2:
            Is a one form differential defined on M in a dictionary format with tuple type 'keys' and string
            type 'values'.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result from calculate {alpha,beta}_P in a dictionary format with tuple type 'keys' and symbol
            type 'values' if latex format is False or in otherwise is the same result but in latex format

        Example
        =======
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # For one form alpha
            >>> one_form_1 = {(1,): 'a1*x1', (2,): 'a2*x2', (3,): 'a3*x3'}
            >>> # For one form beta
            >>> one_form_2 = {(1,): 'b1*x1', (2,): 'b2*x2', (3,): 'b3*x3'}
            >>> # Calculates
            >>> # {alpha,beta}_P = (b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2))*(-x2*x3*dx1 - x1*x3*dx2 - x1*x2*dx3)
            >>> pg.one_forms_bracket(bivector, one_form_1, one_form_2)
            >>> {(1,): -x2*x3*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2)),
                 (2,): -x1*x3*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2)),
                 (3,): -x1*x2*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2))}
            >>> pg.one_forms_bracket(bivector, one_form_1, one_form_2, latex_format=True)
            >>> 'x_{2} x_{3} \\left(a_{1} b_{2} - a_{1} b_{3} - a_{2} b_{1} + a_{2} b_{3} + a_{3} b_{1} -
                 a_{3} b_{2}\\right) \\boldsymbol{dx}_{1} + x_{1} x_{3} \\left(a_{1} b_{2} - a_{1} b_{3} -
                 a_{2} b_{1} + a_{2} b_{3} + a_{3} b_{1} - a_{3} b_{2}\\right) \\boldsymbol{dx}_{2} +
                 x_{1} x_{2} \\left(a_{1} b_{2} - a_{1} b_{3} - a_{2} b_{1} + a_{2} b_{3} + a_{3} b_{1} -
                 a_{3} b_{2}\\right) \\boldsymbol{dx}_{3}'
        """
        # Defines as alpha = one_form_1, beta = one_form_2 and P = bivector.
        # This block converts strings to symbolic variables
        for i in range(1, self.dim + 1):
            one_form_1[(i,)] = sym.sympify(one_form_1.get((i,), 0))
            one_form_2[(i,)] = sym.sympify(one_form_2.get((i,), 0))

        # List with 'dim' zeros
        ii_sharp_alpha_d_beta = [0] * self.dim
        ii_sharp_beta_d_alpha = [0] * self.dim
        """ This block calculates i_P#(alpha)(d_beta) and i_P#(beta)(d_alpha) as
            i_P#(alpha)(d_beta)=P#(alpha)j*(Dxj(bta_i)-Dxi(bta_j))*dxi+P#(alpha)i*(Dxi(bta_j)-Dxj(bta_i))dxj
            for i<j. Is analogous to i_P#(beta)(d_alpha)
        """
        for z in itools.combinations(range(1, self.dim + 1), 2):
            ii_sharp_alpha_d_beta[z[0] - 1] = sym.simplify(
                ii_sharp_alpha_d_beta[z[0] - 1] + self.sharp_morphism(
                    bivector, one_form_1, remove_zeros=False)[(z[1],)] * (sym.diff(
                        one_form_2[(z[0],)], self.coords[z[1] - 1]) - sym.diff(
                        one_form_2[(z[1],)], self.coords[z[0] - 1])))
            ii_sharp_alpha_d_beta[z[1] - 1] = sym.simplify(
                ii_sharp_alpha_d_beta[z[1] - 1] + self.sharp_morphism(
                    bivector, one_form_1, remove_zeros=False)[(z[0],)] * (sym.diff(
                        one_form_2[(z[1],)], self.coords[z[0] - 1]) - sym.diff(
                        one_form_2[(z[0],)], self.coords[z[1] - 1])))
            ii_sharp_beta_d_alpha[z[0] - 1] = sym.simplify(
                ii_sharp_beta_d_alpha[z[0] - 1] + self.sharp_morphism(
                    bivector, one_form_2, remove_zeros=False)[(z[1],)] * (sym.diff(
                        one_form_1[(z[0],)], self.coords[z[1] - 1]) - sym.diff(
                        one_form_1[(z[1],)], self.coords[z[0] - 1])))
            ii_sharp_beta_d_alpha[z[1] - 1] = sym.simplify(
                ii_sharp_beta_d_alpha[z[1] - 1] + self.sharp_morphism(
                    bivector, one_form_2, remove_zeros=False)[(z[0],)] * (sym.diff(
                        one_form_1[(z[1],)], self.coords[z[0] - 1]) - sym.diff(
                        one_form_1[(z[0],)], self.coords[z[1] - 1])))

        """ This block calculate d_P(alpha,beta) that is the same to d(<beta,P#(alpha)>), where <,> the pairing and
            d(<beta,P#(alpha)>) = d(P#(alpha)^i * beta_i)
        """
        d_pairing_beta_sharp_alpha = sym.simplify(sym.derive_by_array(sum(
            value * self.sharp_morphism(bivector, one_form_1, remove_zeros=False).get(ky, 0) for ky, value
            in one_form_2.items()), self.coords))

        # List for the coefficients of {alpha,beta}_P,
        one_forms_brack_aux = []
        for i in range(self.dim):
            one_forms_brack_aux.append(sym.simplify(
                ii_sharp_alpha_d_beta[i] - ii_sharp_beta_d_alpha[i]
                + d_pairing_beta_sharp_alpha[i]))

        # Converts one_forms_brack_aux to dictionary
        keys_list = [(index,) for index in range(1, self.dim + 1)]
        one_forms_brack = dict(zip(keys_list,
                                   one_forms_brack_aux))

        # Return a symbolic type expression or the same expression in latex format
        if latex_format:
            return symbolic_expression(one_forms_brack, self.dim, self.coords, self.variable, dx=True).Mv_latex_str()
        else:
            return remove_values_zero(one_forms_brack)

    def linear_normal_form_R3(self, bivector, latex_format=False):
        """ Calculates a normal form for Lie-Poisson bivector fields on R^3 modulo linear isomorphisms.

        Parameters
        ==========
        :bivector:
            Is a Lie-Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is a normal form in a dictionary format with integer type 'keys' and symbol
            type 'values'.

        Example
        =======
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x1'}
            >>> # Calculates a normal form
            >>> pg.linear_normal_form_R3(bivector)
            >>> {(1,3): 4*a*x2 + x1, (2,3): 4*a*x1 + x2}
            >>> pg.linear_normal_form_R3(bivector, latex_format=True)
            >>> '\\left ( 4 a x_{2} + x_{1}\\right ) \\boldsymbol{Dx}_{1}\\wedge \\boldsymbol{Dx}_{3} +
                 \\left ( 4 a x_{1} + x_{2}\\right ) \\boldsymbol{Dx}_{2}\\wedge \\boldsymbol{Dx}_{3}'
        """
        # Trivial case
        if all(vl == 0 for vl in bivector.values()):
            if latex_format:
                return symbolic_expression({0: 0}, self.dim, self.coords, self.variable).Mv_latex_str()
            else:
                return {0: 0}
        # Converts strings to symbolic variables
        for key in bivector:
            bivector[key] = sym.sympify(bivector[key])
        # List for coords (x1,x2,x3) on R^3
        x_aux = sym.symbols(f'{self.variable}1:{4}')
        # Remember that E = x1*Dx1 + x2*Dx2 + x3*Dx3 (Euler vector field on R3) and P = bivector
        pairing_E_P = sym.simplify(x_aux[0] * bivector.get((2, 3), 0) + (-1) * x_aux[1]
                                   * bivector.get((1, 3), 0) + x_aux[2] * bivector.get((1, 2), 0))
        # Calculates the Hessian matrix of the pairing <E,P>
        hess_pairing_E_P = sym.simplify(sym.derive_by_array(sym.derive_by_array(
            pairing_E_P, x_aux), x_aux)).tomatrix()
        # Diagonalize the Hessian matrix of the pairing <E,P>
        (trnsf, diag_hess) = hess_pairing_E_P.diagonalize()

        """ Classification:
            Unimodular case. The modular vector field of P relative to the Euclid
            volume form on R^3 is zero.
        """
        if all(vl == 0 for vl in self.modular_vf(bivector, 1).values()):
            if hess_pairing_E_P.rank() == 1:
                if latex_format:
                    return symbolic_expression({(2, 3): x_aux[0]}, self.dim, self.coords, self.variable).Mv_latex_str()
                else:
                    return {(2, 3): x_aux[0]}
            if hess_pairing_E_P.rank() == 2:
                # Case: Hessian of <E,P> with index 2
                if diag_hess[0, 0] * diag_hess[1, 1] > 0 or \
                   diag_hess[0, 0] * diag_hess[2, 2] > 0 or \
                   diag_hess[1, 1] * diag_hess[2, 2] > 0:
                    if latex_format:
                        return symbolic_expression({(1, 3): -x_aux[1], (2, 3): x_aux[0]},
                                                   self.dim,
                                                   self.coords,
                                                   self.variable).Mv_latex_str()
                    else:
                        return {(1, 3): -x_aux[1], (2, 3): x_aux[0]}
                else:
                    if latex_format:
                        return symbolic_expression({(1, 3): x_aux[1], (2, 3): x_aux[0]},
                                                   self.dim,
                                                   self.coords,
                                                   self.variable).Mv_latex_str()
                    else:
                        # Case: Hessian of <E,P> with index 1
                        return {(1, 3): x_aux[1], (2, 3): x_aux[0]}
            if hess_pairing_E_P.rank() == 3:
                # Distinguish indices of the Hessian of <E,P>
                index_hess_sum = diag_hess[0, 0]/abs(diag_hess[0, 0]) \
                                 + diag_hess[1, 1]/abs(diag_hess[1, 1]) \
                                 + diag_hess[2, 2]/abs(diag_hess[2, 2])
                if index_hess_sum == 3 or index_hess_sum == -3:
                    if latex_format:
                        return symbolic_expression({(1, 2): x_aux[2], (1, 3): -x_aux[1], (2, 3): x_aux[0]},
                                                   self.dim,
                                                   self.coords,
                                                   self.variable).Mv_latex_str()
                    else:
                        return {(1, 2): x_aux[2], (1, 3): -x_aux[1], (2, 3): x_aux[0]}
                else:
                    if latex_format:
                        return symbolic_expression({(1, 2): -x_aux[2], (1, 3): -x_aux[1], (2, 3): x_aux[0]},
                                                   self.dim,
                                                   self.coords,
                                                   self.variable).Mv_latex_str()
                    else:
                        return {(1, 2): -x_aux[2], (1, 3): -x_aux[1], (2, 3): x_aux[0]}
        # Non-unimodular case
        else:
            if hess_pairing_E_P.rank() == 0:
                if latex_format:
                    return symbolic_expression({(1, 3): x_aux[0], (2, 3): x_aux[1]},
                                               self.dim,
                                               self.coords,
                                               self.variable).Mv_latex_str()
                else:
                    return {(1, 3): x_aux[0], (2, 3): x_aux[1]}
            if hess_pairing_E_P.rank() == 2:
                # Case: Hessian of <E,P> with index 2
                if diag_hess[0, 0] * diag_hess[1, 1] > 0 or \
                        diag_hess[0, 0] * diag_hess[2, 2] > 0 or \
                        diag_hess[1, 1] * diag_hess[2, 2] > 0:
                    if latex_format:
                        return symbolic_expression({(1, 3): x_aux[0] - 4 * sym.sympify('a') * x_aux[1],
                                                    (2, 3): x_aux[1] + 4 * sym.sympify('a') * x_aux[0]},
                                                   self.dim,
                                                   self.coords,
                                                   self.variable).Mv_latex_str()
                    else:
                        return {(1, 3): x_aux[0] - 4 * sym.sympify('a') * x_aux[1],
                                (2, 3): x_aux[1] + 4 * sym.sympify('a') * x_aux[0]}
                else:
                    if latex_format:
                        return symbolic_expression({(1, 3): x_aux[0] + 4 * sym.sympify('a') * x_aux[1],
                                                    (2, 3): 4 * sym.sympify('a') * x_aux[0] + x_aux[1]},
                                                   self.dim,
                                                   self.coords,
                                                   self.variable).Mv_latex_str()
                    else:
                        # Case: Hessian of <E,P> with index 1
                        return {(1, 3): x_aux[0] + 4 * sym.sympify('a') * x_aux[1],
                                (2, 3): 4 * sym.sympify('a') * x_aux[0] + x_aux[1]}
            if hess_pairing_E_P.rank() == 1:
                if latex_format:
                    return symbolic_expression({(1, 3): x_aux[0], (2, 3): 4 * x_aux[0] + x_aux[1]},
                                               self.dim,
                                               self.coords,
                                               self.variable).Mv_latex_str()
                else:
                    return {(1, 3): x_aux[0], (2, 3): 4 * x_aux[0] + x_aux[1]}

    def isomorphic_lie_poisson_R3(self, bivector_1, bivector_2):
        """ Determines if two Lie-Poisson bivector fields on R^3 are isomorphic or not.

        Parameters
        ==========
        :bivector_1/bivector_2:
            Is a Lie-Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if the bivectors are isomorphic, in other case is False.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector_1 = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # For bivector x1*Dx2^Dx3
            >>> bivector_2 = {(1,2): 'x1', (1,3): '0', (2,3): '0'}
            >>> isomorphic_lie_poisson_R3(bivector_1, bivector_2)
            >>> False
        """
        return True if self.linear_normal_form_R3(bivector_1) == self.linear_normal_form_R3(bivector_2) else False

    def gauge_transformation(self, bivector, two_form):
        """ This method compute the Gauge transformation of a Poisson bivector field.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :two_form:
            Is a closed differetial form in a dictionary format with tuple type 'keys' and string type 'values'.

        Returns
        =======
            A poisson bivector filed wich is the Gauge transformation relative to the two_form param of the given
            Poisson bivector field in a dictionary format with integer type 'keys' and symbol
            type 'values'.
            The determinat from the matrix C in a matrix symbol format, where C = I-A*B with A the matrix
            asociated from the bivector and B the matrix asociated from the two_form.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector P12*Dx1^Dx2 + P13*Dx1^Dx3 + P23*Dx2^Dx3
            >>> P = {(1,2): ‘P12’, (1,3): ‘P13’, (2,3): ‘P23’}
            >>> # For two form L12*dx1^dx2 + L13*dx1^dx3 + L23*dx2^dx3
            >>> lambda = {(1,2): ‘L12’, (1,3): ‘L13’, (2,3): ‘L23’}
            >>> (gauge bivector, determinant) = pg.gauge transformation(P, lambda)
            >>> gauge bivector
            >>> {  (1, 2): P12/(L12*P12 + L13*P13 + L23*P23 + 1),
                   (1, 3): P13/(L12*P12 + L13*P13 + L23*P23 + 1),
                   (2, 3): P23/(L12*P12 + L13*P13 + L23*P23 + 1)
                }
            >>> determinant
            >>> (L12*P12 + L13*P13 + L23*P23 + 1)**2
        """
        two_form_B = two_tensor_form_to_matrix(two_form, self.dim)
        I_minus_BP = sym.Matrix(sym.simplify(sym.eye(self.dim) - two_form_B * self.bivector_to_matrix(bivector)))
        det_I_minus_BP = sym.factor(sym.simplify(I_minus_BP.det()))
        if det_I_minus_BP == 0:
            return False
        else:
            gauge_mtx = sym.Matrix(sym.simplify(self.bivector_to_matrix(bivector) * (I_minus_BP.inv())))
        gauge_biv = {}
        for z in itools.combinations(range(1, self.dim + 1), 2):
            gauge_biv.setdefault(z, gauge_mtx[z[0]-1, z[1]-1])
        return gauge_biv, sym.sympify(det_I_minus_BP)

    def flaschka_ratiu_bivector(self, casimirs, symplectic_form=False, latex_format=False):
        """ Calculate a Poisson bivector from Flaschka-Ratui formula where all Casimir function is in "casimir"
        variable. This Poisson bivector is the following form:
            i_π Ω := dK_1^...^dK_m-2
        where K_1, ..., K_m-2 are casimir functions and (M, Ω) is a diferentiable manifold with volument form Ω
        and dim(M)=m.

        Parameters
        ==========
        :casimirs:
            Is a list of Casimir functions where each element is a string type.
        :symplectic_form:
            Is a boolean value to indicates if is necesarry calculate the symplectic form relative to Poisson
            bivector, its default value is False
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value
            is False

        Returns
        =======
            The result is a Poisson bivector with Casimir function casimirs in a dictionary format with tuple
            type 'keys' and symbol type 'values' if latex format is False or in otherwise is the same result but
            in latex format. Also if symplectic_form is False is it not calculate in otherwise is calculated.

        Example
        ========
            >>> # Instance the class to dimension 4
            >>> pg = PoissonGeometry(4)
            >>> casimirs = ['x1**2 +x2**2 +x3**2', 'x4']
            >>> pg.flaschka_ratiu_bivector(casimirs, symplectic_form=False, latex_format=False)
            >>> {(1, 2): -2*x3, (1, 3): 2*x2, (2, 3): -2*x1}
            >>> pg.flaschka_ratiu_bivector(casimirs, symplectic_form=False, latex_format=True)
            >>> '-2*x3*Dx1^Dx2 + 2*x2*Dx1^Dx3 - 2*x1*Dx2^Dx3'
            >>> bivector, symplectic_form = pg.flaschka_ratiu_bivector(casimirs, symplectic_form=True,
                                                                       latex_format=False)
            >>> bivector
            >>> {(1, 2): -2*x3, (1, 3): 2*x2, (1, 4): 0, (2, 3): -2*x1, (2, 4): 0, (3, 4): 0}
            >>> symplectic_form
            >>> x3*Dx1^Dx2/(2*(x1**2 + x2**2 + x3**2)) - x2*Dx1^Dx3/(2*x1**2 + 2*x2**2 + 2*x3**2) + x1*Dx2^Dx3/(2*(x1**2 + x2**2 + x3**2)  # noqa
            >>> bivector, symplectic_form = pg.flaschka_ratiu_bivector(casimirs, symplectic_form=True, latex_format=True)  # noqa
            >>> bivector
            >>> '- 2 x_{3} \\boldsymbol{Dx}_{1}\\wedge \\boldsymbol{Dx}_{2} + 2 x_{2} \\boldsymbol{Dx}_{1}
                 \\wedge \\boldsymbol{Dx}_{3} - 2 x_{1} \\boldsymbol{Dx}_{2}\\wedge \\boldsymbol{Dx}_{3}'
            >>> symplectic_form
            >>> '\\frac{x_{3}}{2 \\left({\\left ( x_{1} \\right )}^{2} + {\\left ( x_{2} \\right )}^{2} +
                 {\\left ( x_{3} \right )}^{2}\\right)} \\boldsymbol{Dx}_{1}\\wedge \\boldsymbol{Dx}_{2}
                 - \\frac{x_{2}}{2 {\\left ( x_{1} \\right )}^{2} + 2 {\\left ( x_{2} \\right )}^{2} + 2
                 {\\left ( x_{3} \right )}^{2}} \\boldsymbol{Dx}_{1}\\wedge \boldsymbol{Dx}_{3} +
                 \frac{x_{1}}{2 \\left({\\left ( x_{1} \\right )}^{2} + {\\left ( x_{2} \\right )}^{2} +
                 {\\left ( x_{3} \right )}^{2}\\right)} \\boldsymbol{Dx}_{2}\\wedge \\boldsymbol{Dx}_{3}'
        """

        if self.dim - len(casimirs) == 2:
            # Convert all Casimir functions in sympy objects
            casimir_list = [sym.sympify(casimir) for casimir in casimirs]

            # This block calculates d(C), where C is a function Casimir
            diff_casimir_list = []
            diff_casimir_matrix = []
            for casimir in casimir_list:
                casimir_matrix = []
                diff_casimir = 0
                for i, variable in enumerate(self.coords):
                    casimir_matrix.append(casimir.diff(variable))
                    diff_casimir = diff_casimir + (casimir.diff(variable) * self.Dx_basis[i])
                diff_casimir_matrix.append(casimir_matrix)
                diff_casimir_list.append(diff_casimir)

            # This block calculates d(C_1)^...^d(C_dimension)
            d_casimir = diff_casimir_list[0]
            for index, element in enumerate(diff_casimir_list):
                d_casimir = d_casimir ^ element if index != 0 else d_casimir

            if d_casimir.is_zero():
                return {0: 0}

            # This blocks obtains Poisson coefficients
            bivector_coeff_dict = {}
            combinations = [x for x in itools.combinations(range(1, self.dim + 1), 2)]
            for combination in combinations:
                combination = list(combination)
                casimir_matrix_without_combination = sym.Matrix(diff_casimir_matrix)
                for i, element in enumerate(combination):
                    casimir_matrix_without_combination.col_del(element - (i + 1))

                # Makes a dictionary with Poisson coefficients
                key = (combination[0], combination[1])
                coeff = ((-1)**(combination[0] + combination[1]))
                bivector_coeff_dict[key] = sym.simplify(coeff * casimir_matrix_without_combination.det())

            if symplectic_form:
                # Makes a basis {dxi^dxj}_{i<j} and it save in a dictionary type variable
                dx_ij_basis = {}
                i = 0
                while i < len(self.Dx_basis):
                    j = i + 1
                    while j < len(self.Dx_basis):
                        key = int(F"{i+1}{j+1}")
                        dx_ij_basis[key] = self.Dx_basis[i] ^ self.Dx_basis[j]
                        j = j + 1
                    i = i + 1

                bivector = 0
                estructure_symplectic_num = 0
                estructure_symplectic_dem = 0
                for key in dx_ij_basis.keys():
                    i, j = int(str(key)[0]), int(str(key)[1])
                    bivector = bivector + bivector_coeff_dict[(i, j)] * dx_ij_basis[key]
                    estructure_symplectic_num = estructure_symplectic_num + bivector_coeff_dict[(i, j)] * dx_ij_basis[key] # noqa
                    estructure_symplectic_dem = estructure_symplectic_dem + bivector_coeff_dict[(i, j)] * bivector_coeff_dict[(i,j)] # noqa

                symplectic = estructure_symplectic_num * (-1 / estructure_symplectic_dem)

                if latex_format:
                    return [symbolic_expression(bivector_coeff_dict,
                                                self.dim,
                                                self.coords,
                                                self.variable).Mv_latex_str(),
                            symplectic.Mv_latex_str()]
                else:
                    return [bivector_coeff_dict, symplectic]
            else:
                return sym.latex(symbolic_expression(bivector_coeff_dict, self.dim, self.coords, self.variable)) if latex_format else remove_values_zero(bivector_coeff_dict) # noqa
        else:
            return {0: 0}
