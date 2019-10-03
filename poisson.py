"""
 Copyright 2019 by P Suarez-Serrato, Jose Ruíz and M Evangelista-Alvarado.
 Instituto de Matematicas (UNAM-CU) Mexico
 This is free software; you can redistribute it and/or
 modify it under the terms of the MIT License,
 https://en.wikipedia.org/wiki/MIT_License.

 This software has NO WARRANTY, not even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
from __future__ import unicode_literals

import string
import sympy as sym
import itertools as itools
from utils import (validate_dimension, symbolic_expression,
                   is_dicctionary, show_coordinates,
                   two_tensor_form_to_matrix,
                   is_string, basis, symbolic_expression_2, values_to_kernel,
                   derivate_formal, derivate_standar, symbolic_multivector_to_dict)


class PoissonGeometry:
    """ This class provides some useful tools for Poisson-Nijenhuis calculus on Poisson manifolds."""

    def __init__(self, dimension, variable='x'):
        # TODO coordinates cambiar por coords #
        # Obtains the dimension
        self.dim = validate_dimension(dimension)
        # Define what variables the class will work with
        self.variable = variable
        # Create the symbolics symbols
        self.coordinates = sym.symbols(f'{self.variable}1:{self.dim + 1}')
        # Show the coordinates with that will the class works
        self.coords = show_coordinates(self.coordinates)
        self.Dx_basis = basis(self.dim, self.coordinates, self.variable)

    def bivector_to_matrix(self, bivector):
        """ Constructs the matrix of a 2-contravariant tensor field or bivector field.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is a symbolic skew-symmetric matrix of dimension (dim)x(dim) if latex_format is False
            in otherwise the result is the same but in latex format.

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> bivector_to_matrix(bivector, latex_format=False)
            >>> Matrix([[0, x3, -x2], [-x3, 0, x1], [x2, -x1, 0]])
            >>> bivector_to_matrix(bivector, latex_format=True)
            >>> ''
        """
        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
        except Exception as e:
            print(F"[Error bivector_to_matrix] \n Error: {e}, Type: {type(e)}")

        # Creates a symbolic Matrix
        bivector_matrix = sym.MatrixSymbol('P', self.dim, self.dim)
        bivector_matrix = sym.Matrix(bivector_matrix)

        # Assigns the corresponding coefficients of the bivector field
        for ij in itools.combinations_with_replacement(range(1, self.dim + 1), 2):
            i, j = ij[0], ij[1]
            # Makes the Poisson Matrix
            bivector_matrix[i - 1, j - 1] = 0 if i == j else sym.sympify(bivector[int(''.join(str(i) for i in ij))])
            bivector_matrix[j - 1, i - 1] = (-1) * bivector_matrix[i - 1, j - 1]

        # Return a symbolic Poisson matrix or the same expression in latex format
        return bivector_matrix

    def sharp_morphism(self, bivector, one_form, latex_format=False):
        """ Calculates the image of a differential 1-form under the vector bundle morphism 'sharp' P#: T*M -> TM
            defined by P#(alpha) := i_(alpha)P, where P is a Poisson bivector field on a manifold M, alpha a 1-form
            on M and i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :one_form:
            Is a 1-form differential in a dictionary format with integer type 'keys' and string type 'values'.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is the image of a differential 1-form alpha under the vector bundle morphism 'sharp' in a
            dictionary format with integer type 'keys' and symbol type 'values'

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For one form a1*x1*dx1 + a2*x2*dx2 + a3*x3*dx3.
            >>> one_form = {1: 'a1*x1', 2: 'a2*x2', 3: 'a3*x3'}
            >>> # P#(one_form) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
            >>> sharp_morphism(bivector, one_form, latex_format=False)
            >>> {1: x2*(-a2*x3 + a3*x3x), 2: x1*(a1*x3 - a3*x3x), 3: x1*x2*(-a1 + a2)}
            >>> sharp_morphism(bivector, one_form, latex_format=True)
            >>> x_{2} x_{3} \\left(- a_{2} + a_{3}\\right) \\boldsymbol{Dx}_{1} + x_{1} x_{3} \\left
                (a_{1} - a_{3}\\right) \\boldsymbol{Dx}_{2} + x_{1} x_{2} \\left(- a_{1} + a_{2}\right)
                \\boldsymbol{Dx}_{3}
        """

        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
            one_form = is_dicctionary(one_form)
        except Exception as e:
            print(F"[Error sharp_morphism] \n Error: {e}, Type: {type(e)}")

        # Converts strings to symbolic variables
        for i, coeff_i in one_form.items():
            one_form.update({i: sym.sympify(coeff_i)})
        for ij, coeff_ij in bivector.items():
            bivector.update({ij: sym.sympify(coeff_ij)})

        """
            Calculation of the vector field bivector^#(one_form) as
                ∑_{i}(∑_{j} bivector_ij (one_form_i*Dx_j - one_form_j*Dx_i)
            where i < j.
        """
        p_sharp_aux = [0] * self.dim
        for ij, bivector_ij in bivector.items():
            # Get the values i and j from bivector index ij
            i, j = int(str(ij)[0]), int(str(ij)[1])
            # Calculates one form i*Pij*Dxj
            p_sharp_aux[j - 1] = sym.simplify(p_sharp_aux[j - 1] + one_form[i] * bivector_ij)
            # Calculates one form j*Pji*Dxi
            p_sharp_aux[i - 1] = sym.simplify(p_sharp_aux[i - 1] - one_form[j] * bivector_ij)
        # Creates a dictionary which represents the vector field P#(alpha)
        p_sharp = dict(zip(range(1, self.dim + 1), p_sharp_aux))

        # Return a vector field expression in LaTeX format
        if latex_format:
            # For copy and paste this result in a latex editor, do not forget make "print()" in the result.
            return symbolic_expression(p_sharp, self.dim, self.coordinates, self.variable).Mv_latex_str()
        # Return a symbolic dictionary.
        return p_sharp

    def is_in_kernel(self, bivector, one_form):
        """ Check if a differential 1-form alpha belongs to the kernel of a given Poisson bivector field,
            that is check if P#(alpha) = 0

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :one_form:
            Is a 1-form differential in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if P#(alpha) = 0, in other case is False.

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For one form x1*dx1 + x2*dx2 + x3*dx3.
            >>> one_form = {1: 'x1', 2: 'x2', 3: 'x3'}
            >>> # P#(one_form) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
            >>> is_in_kernel(bivector, one_form)
            >>> True
        """
        p_sharp = self.sharp_morphism(bivector, one_form)
        # Converts a dictionary symbolic to a symbolic expression and verify is zero with a sympy method
        return True if symbolic_expression(p_sharp, self.dim, self.coordinates, self.variable).is_zero() else False

    def hamiltonian_vector_field(self, bivector, hamiltonian_function, latex_format=False):
        """ Calculates the Hamiltonian vector field of a function relative to a Poisson bivector field as follows:
            X_h = P#(dh), where d is the exterior derivative of h and P#: T*M -> TM is the vector bundle morphism
            defined by P#(alpha) := i_(alpha)P, with i is the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :hamiltonian_function:
            Is a function scalar h: M --> R that is a string type.
        :latex_format:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is the hamiltonian vector field relative to a Poisson in a dictionary format with integer type
            'keys' and symbol type 'values'

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For hamiltonian_function h(x1,x2,x3) = x1 + x2 + x3.
            >>> hamiltonian_function = 'x1 + x2 + x3'
            >>> # X_h = P#(dh) = (x2 - x3)*Dx1 + (-x1 + x3)*Dx2 + (x1 - x2)*Dx3.
            >>> hamiltonian_vector_field(bivector, hamiltonian_function, latex_format=False)
            >>> {1: x2 - x3, 2: -x1 + x3, 3: x1 - x2}
            >>> hamiltonian_vector_field(bivector, hamiltonian_function, latex_format=True)
            >>> \\left ( x_{2} - x_{3}\\right ) \\boldsymbol{Dx}_{1} + \\left ( - x_{1} + x_{3}\\right )
                \\boldsymbol{Dx}_{2} + \\left ( x_{1} - x_{2}\\right ) \\boldsymbol{Dx}_{3}'
        """

        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
            hamiltonian_function = is_string(hamiltonian_function)
        except Exception as e:
            print(F"[Error hamiltonian_vector_field] {e}, Type: {type(e)}")

        # Converts a string to symbolic expression
        h = sym.sympify(hamiltonian_function)
        # Calculates the differential of hamiltonian_function
        dh = sym.derive_by_array(h, self.coordinates)
        # Calculates the Hamiltonian vector field
        hamiltonian_vector_field = self.sharp_morphism(bivector, dict(zip(range(1, self.dim + 1), dh)))

        if latex_format:
            # return to symbolic expression in LaTeX format
            return symbolic_expression(hamiltonian_vector_field, self.dim,
                                       self.coordinates, self.variable).Mv_latex_str()
        else:
            # return a symbolic dictionary
            return hamiltonian_vector_field

    def is_casimir(self, bivector, function):
        """ Check if a function is a Casimir function of a given (Poisson) bivector field, that is
            P#(dK) = 0, where dK is the exterior derivative (differential) of K and P#: T*M -> TM is the
            vector bundle morphism defined by P#(alpha) := i_(alpha)P, with i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :function:
            Is a function scalar f: M --> R that is a string type.

        Returns
        =======
            The result is True if P#(df) = 0, in other case is False.

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For function f(x1,x2,x3) = x1^2 + x2^2 + x3^2.
            >>> function = 'x1**2 + x2**2 + x3**2'
            >>> is_casimir(bivector, function)
            >>> True
        """

        # Calculates the Hamiltonian vector field
        hamiltonian_vector_field = self.hamiltonian_vector_field(bivector, function)
        # Converts a dictionary symbolic to a symbolic expression and verify is zero with a method of sympy
        return True if symbolic_expression(hamiltonian_vector_field, self.dim,
                                           self.coordinates, self.variable).is_zero() else False

    def poisson_bracket(self, bivector, f, g, latex_format=False):
        """ Calculates the poisson bracket {f,g} = π(df,dg) = ⟨dg,π#(df)⟩ of two functions
            f and g on a Poisson manifold (M,P), where d is the exterior derivatives and P#: T*M -> TM is the
            vector bundle morphism defined by P#(alpha) := i_(alpha)P, with i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :f / g:
            Is a function scalar f/g: M --> R that is a string type.

        Returns
        =======
            The result is the poisson bracket from f with g in a dictionary format with integer type 'keys' and
            symbol type 'values' if latex format is False or in otherwise is the same result but in latex format

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For f(x1,x2,x3) = x1 + x2 + x3.
            >>> f = 'x1 + x2 + x3'
            >>> # For g(x1,x2,x3) = 'a1*x1 + a2*x2 + a3*x3'.
            >>> g = 'a1*x1 + a2*x2 + a3*x3'
            >>> # {f, g} = a1*(x2 - x3) - a2*(x1 - x3) + a3*(x1 - x2)
            >>> poisson_bracket(bivector, f, g, latex_format=False)
            >>> a1*(x2 - x3) - a2*(x1 - x3) + a3*(x1 - x2)
            >>> poisson_bracket(self, bivector, f, g, latex_format=True)
            >>> 'a_{1} \\left(x_{2} - x_{3}\\right) - a_{2} \\left(x_{1} - x_{3}\\right) + a_{3} \\left(x_{1}
                - x_{2}\\right)'
        """

        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
            f = is_string(f)
            g = is_string(g)
        except Exception as e:
            print(F"[Error poisson_bracket] \n Error: {e}, Type: {type(e)}")

        # Convert from string to sympy value
        f, g = sym.sympify(f), sym.sympify(g)
        if f == g:
            # {f,g} = 0 if f=g
            return 0
        # Calculates the differentials of f and g
        df = dict(zip(range(1, self.dim + 1), sym.derive_by_array(f, self.coordinates)))
        dg = dict(zip(range(1, self.dim + 1), sym.derive_by_array(g, self.coordinates)))
        # Calculates {f,g} = <g,P#(df)> = ∑_{i} (P#(df))^i * (dg)_i
        bracket_f_g = sym.simplify(sum(self.sharp_morphism(bivector, df)[index] * dg[index] for index in df))

        # Return a symbolic type expression or the same expression in latex format
        return sym.latex(bracket_f_g) if latex_format else bracket_f_g

    def schouten_bracket(self, A, B, latex_format=False):
        """"""
        # Obtains the dimesion to multivectors A and B
        dim_A = 0 if isinstance(A, str) else len(str(list(A.keys())[0]))
        dim_B = 0 if isinstance(B, str) else len(str(list(B.keys())[0]))

        # Converts to sympy variable the params A and B
        for param in [A, B]:
            if isinstance(param, str):
                pass
            else:
                for index, coeff_index in param.items():
                    param.update({index: sym.sympify(coeff_index)})

        # Calculates the formula
        index = 1
        result = 0
        factor = -(-1)**((dim_A-1)*(dim_B-1))
        while index < self.dim + 1:
            # Calculates dA/de ^ dB/dx
            dA_formal = symbolic_expression_2(derivate_formal(A, index), self.Dx_basis)
            dB_standar = symbolic_expression_2(derivate_standar(B, self.coordinates[index - 1]), self.Dx_basis)
            dA_f_wedge_dB_s = dA_formal ^ dB_standar
            # Calculates dB/de ^ dA/dx
            dB_formal = symbolic_expression_2(derivate_formal(B, index), self.Dx_basis)
            dA_standar = symbolic_expression_2(derivate_standar(A, self.coordinates[index - 1]), self.Dx_basis)
            dB_f_wedge_dA_s = int(factor) * dB_formal ^ dA_standar
            result = result + dA_f_wedge_dB_s + dB_f_wedge_dA_s
            index = index + 1
        # Return a symbolic type expression or the same expression in latex format
        if latex_format:
            return sym.latex(result)
        else:
            result_dict = symbolic_multivector_to_dict(result, self.dim)
            # clean the result_dict
            clean_result_dict = {}
            for key, value in result_dict.items():
                if len(key) == (dim_A+dim_B-1):
                    clean_result_dict.update({int(key): value})
            return clean_result_dict

    def cohomology_lineal_gruops(self, multivector):
        """"""
        cohomology_lineal_gruops = {}
        i = 1
        alphabet = string.ascii_lowercase
        while i < self.dim + 1:
            print(F'Calculating to i = {i}')
            if i == 1:
                # Image
                cocicle_imagen_list = [f'a{i+1}*{value}' for i, value in enumerate(self.coordinates)]
                cocicle_imagen_str = "+".join(cocicle_imagen_list)
                imagen = self.schouten_bracket(multivector, cocicle_imagen_str)
                print(F'Imagen: {imagen}')
                # genetares basis
                variable_with_zero_value = {}
                cocicle_imagen_list = [f'a{i+1}' for i, value in enumerate(self.coordinates)]
                for ai in cocicle_imagen_list:
                    variable_with_zero_value.update({ai: 0})
                imagen_basis = {}
                for index in range(0, len(cocicle_imagen_list)):
                    variable_with_zero_value.update({F'a{index+1}': 1})
                    expresion = symbolic_expression_2(imagen, self.Dx_basis)
                    imagen_basis.update({F'a{index+1}': expresion.subs(variable_with_zero_value)})
                    variable_with_zero_value.update({F'a{index+1}': 0})
                print(F'The basis are: {list(imagen_basis.values())}')

                # kernel
                cocicle_kernel_values = {}
                lineal_function_variables = []
                for index, value in enumerate(self.Dx_basis):
                    lineal_function_to_dxi_list = [f'{alphabet[index]}{i+1}*{value}' for i, value in enumerate(self.coordinates)] # noqa
                    lineal_function_to_dxi_str = "+".join(lineal_function_to_dxi_list)
                    # variables to find kernel of cofrontera operator
                    lineal_function_variables = lineal_function_variables + [sym.sympify(f'{alphabet[index]}{i+1}') for i, value in enumerate(self.coordinates)] # noqa
                    cocicle_kernel_values.update({index+1: lineal_function_to_dxi_str})
                multivector_image = self.schouten_bracket(multivector, cocicle_kernel_values)
                kernel = values_to_kernel(multivector_image, lineal_function_variables, self.dim, self.coordinates)
                print(F'kernel: {kernel}')
                # genetares basis
                variable_with_zero_value = {}
                for variable in lineal_function_variables:
                    variable_with_zero_value.update({str(variable): 0})

                cocicle_kernel_expresion = symbolic_expression_2(cocicle_kernel_values, self.Dx_basis).subs(kernel)
                kernel_basis = {}
                for index, value in enumerate(lineal_function_variables):
                    variable_with_zero_value.update({str(value): 1})
                    kernel_basis.update({str(value): cocicle_kernel_expresion.subs(variable_with_zero_value)})
                    variable_with_zero_value.update({str(value): 0})
                print(F'The basis are: {list(kernel_basis.values())}')
                cohomology_lineal_gruops.update({i: {'imagen': list(imagen_basis.values()),
                                                     'kernel': list(kernel_basis.values())}})
            elif i < self.dim+1:
                # Image
                cocicle_image_values = {}
                combinations = itools.combinations(range(1, self.dim + 1), i - 1)
                lineal_funtions_vars = []
                for index, z in enumerate(combinations):
                    lineal_function_to_dxi_list = [f'{alphabet[index]}{i+1}*{value}' for i, value in enumerate(self.coordinates)] # noqa
                    lineal_function_to_dxi_str = "+".join(lineal_function_to_dxi_list)
                    # Obtains all variable to use for calculing Schouten bracket
                    lineal_funtions_vars = lineal_funtions_vars + [f'{alphabet[index]}{i+1}' for i, value in enumerate(self.coordinates)] # noqa
                    cocicle_image_values.update({int(''.join(map(str, list(z)))): lineal_function_to_dxi_str})
                imagen = self.schouten_bracket(multivector, cocicle_image_values)
                print(F'image: {imagen}')

                # genetares basis
                variable_with_zero_value = {}
                for ai in lineal_funtions_vars:
                    variable_with_zero_value.update({ai: 0})
                imagen_basis = {}
                for ai in lineal_funtions_vars:
                    variable_with_zero_value.update({ai: 1})
                    expresion = symbolic_expression_2(imagen, self.Dx_basis)
                    imagen_basis.update({ai: expresion.subs(variable_with_zero_value)})
                    variable_with_zero_value.update({ai: 0})
                print(F'The basis are: {list(imagen_basis.values())}')

                # kernel
                cocicle_kernel_values = {}
                lineal_function_variables = []
                combinations = itools.combinations(range(1, self.dim + 1), i)
                for index, z in enumerate(combinations):
                    lineal_function_to_dxi_list = [f'{alphabet[index]}{i+1}*{value}' for i, value in enumerate(self.coordinates)] # noqa
                    lineal_function_to_dxi_str = "+".join(lineal_function_to_dxi_list)
                    lineal_function_variables = lineal_function_variables + [sym.sympify(f'{alphabet[index]}{i+1}') for i, value in enumerate(self.coordinates)] # noqa
                    cocicle_kernel_values.update({int(''.join(map(str, list(z)))): lineal_function_to_dxi_str})
                multivector_image = self.schouten_bracket(multivector, cocicle_kernel_values)
                print(F'multivector_image: {multivector_image}')
                kernel = values_to_kernel(multivector_image, lineal_function_variables, self.dim, self.coordinates)
                print(F'kernel: {kernel}')
                # genetares basis
                variable_with_zero_value = {}
                for variable in lineal_function_variables:
                    variable_with_zero_value.update({str(variable): 0})

                cocicle_kernel_expresion = symbolic_expression_2(cocicle_kernel_values, self.Dx_basis).subs(kernel)
                kernel_basis = {}
                for index, value in enumerate(lineal_function_variables):
                    variable_with_zero_value.update({str(value): 1})
                    kernel_basis.update({str(value): cocicle_kernel_expresion.subs(variable_with_zero_value)})
                    variable_with_zero_value.update({str(value): 0})
                print(F'The basis are: {list(kernel_basis.values())}')
                cohomology_lineal_gruops.update({i: {'imagen': list(imagen_basis.values()),
                                                     'kernel': list(kernel_basis.values())}})
            else:
                cohomology_lineal_gruops.update({i + 1: {'imagen': 0, 'kernel': 0}})
            i = i + 1
        return cohomology_lineal_gruops

    def lichnerowicz_poisson_operator(self, bivector, multivector, latex_format=False):
        """ Calculates the Schouten-Nijenhuis bracket between a given (Poisson) bivector field and a (arbitrary)
        multivector field.
        Recall that the Lichnerowicz-Poisson operator on a Poisson manifold (M,P) is defined as the adjoint operator
        of P, ad_P, respect to the Schouten bracket for multivector fields on M:
            ad_P(A) := [P,A],
        for any multivector field A on M. Here [,] denote the Schouten-Nijenhuis bracket.
        Let P = Pij*Dxi^Dxj (i < j) and A = A^J*Dxj_1^Dxj_2^...^Dxj_a. Here, we use the multi-index notation
        A^J := A^j_1j_2...j_a for j_1 < j_2 <...< j_a.
        Then,
            [P,A](df1,...,df(a+1)) = sum_(i=1)^(a+1) (-1)**(i)*{fi,A(df1,...,î,...,df(a+1))}_P
                                     + sum(1<=i<j<=a+1) (-1)**(i+j)*A(d{fi,fj}_P,..î..^j..,df(a+1)),
        for any smooth functions fi on M. Where {,}_P is the Poisson bracket induced by P. Here ^ denotes the absence
        of the corresponding index and dfi is the differential of fi.
        :param bivector: is a dictionary with integer type 'keys' and string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
        The 'keys' are the ordered indices of the given bivector field P and the 'values' their coefficients.
        In this case, P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param multivector: is a dictionary with integer type 'keys' and string type 'values'. For example, on R^3,
            {123: 'x1*x2*x3'}.
        The 'keys' are the ordered indices of a given multivector field A and the 'values' their coefficients.
        In this case, A = x1*x2*x3*Dx1^Dx2*Dx3.
        :return: a dictionary with integer type 'keys' and symbol type 'values' or the zero value.
        For the previous example, if latex_format is False is zero and if latex_format is True is '0' which says that
        any 4-multivector field on R^3 is zero.
        """
        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
            multivector = is_dicctionary(multivector)
        except Exception as e:
            print(F"[Error lichnerowicz_poisson_operator] \n Error: {e}, Type: {type(e)}")

        mltv = multivector
        # Degree of multivector
        deg_mltv = len(tuple(str(next(iter(mltv)))))
        if deg_mltv + 1 > self.dim:
            return 0
        else:
            # In this case, multivector is a function
            if isinstance(multivector, str):
                # [P,f] = -X_f, for any function f.
                return self.hamiltonian_vector_field(bivector, (-1)*sym.sympify(multivector))
            else:
                # A dictionary for the first term of [P,A], A a multivector
                schouten_biv_mltv_1 = {}
                #
                lich_poiss_aux_1 = 0
                # The first term of [P,A] is a sum of Poisson brackets {xi,A^J}
                for z in itools.combinations(range(1, self.dim + 1),
                                             deg_mltv + 1):
                    for i in z:
                        # List of indices for A without the index i
                        nw_idx = [e for e in z if e not in [i]]
                        # Calculates the Poisson bracket {xi,A^J}, J == nw_idx
                        lich_poiss_aux_11 = sym.simplify(
                            (-1)**(z.index(i)) * self.poisson_bracket(
                                bivector, self.coordinates[i - 1], sym.sympify(
                                    mltv[int(''.join(str(i) for i in nw_idx)
                                             )])))
                        lich_poiss_aux_1 = sym.simplify(lich_poiss_aux_1 + lich_poiss_aux_11)
                    # Add the brackets {xi,A^J} in the list schouten_biv_mltv_1
                    schouten_biv_mltv_1.setdefault(int(''.join(str(j) for j in z)),
                                                   lich_poiss_aux_1)
                    lich_poiss_aux_1 = 0
                # A dictionary for the second term of [P,A]
                schouten_biv_mltv_2 = {}
                sum_i = 0
                sum_y = 0
                # Scond term of [P,A] is a sum of Dxk(P^iris)*A^((J+k)\iris))
                for z in itools.combinations(range(1, self.dim + 1),
                                             deg_mltv + 1):
                    for y in itools.combinations(z, 2):
                        for i in range(1, self.dim + 1):
                            # List of indices for A without the indices in y
                            nw_idx_mltv = [e for e in z if e not in y]
                            # Add the index i
                            nw_idx_mltv.append(i)
                            # Sort the indices
                            nw_idx_mltv.sort()
                            # Ignore index repetition
                            if nw_idx_mltv.count(i) != 2:
                                sum_i_aux = sym.simplify(
                                    (-1)**(z.index(y[0]) + z.index(y[1])
                                           + nw_idx_mltv.index(i))
                                    * sym.diff(sym.sympify(
                                        bivector[
                                            int(''.join(str(j) for j in y))]),
                                        self.coordinates[i - 1])
                                    * sym.sympify(
                                        mltv[int(''.join(
                                            str(k) for k in nw_idx_mltv))]))
                                sum_i = sym.simplify(sum_i + sum_i_aux)
                        sum_y_aux = sum_i
                        sum_y = sym.simplify(sum_y + sum_y_aux)
                        sum_i = 0
                        sum_i_aux = 0
                    # Add the terms Dxk(P^iris)*A^((J+k)\iris)) in the list schouten_biv_mltv_2
                    schouten_biv_mltv_2.setdefault(
                        int(''.join(str(j) for j in z)), sum_y)
                    sum_y = 0
                    sum_y_aux = 0
                # A dictionary for the the Schouten bracket [P,A]
                schouten_biv_mltv = {}
                # Sum and add the terms in schouten_biv_mltv_1 and schouten_biv_mltv_2
                for ky in schouten_biv_mltv_1:
                    schouten_biv_mltv.setdefault(
                        ky, sym.simplify(schouten_biv_mltv_1[ky] + schouten_biv_mltv_2[ky]))
                # return a dictionary or latex format.
                if latex_format:
                    return symbolic_expression(schouten_biv_mltv, self.dim, self.coordinates).Mv_latex_str()
                else:
                    return schouten_biv_mltv

    def jacobiator(self, bivector, latex_format=False):
        """ Calculates de Schouten-Nijenhuis bracket of a given bivector field with himself, that is [P,P]
            where [,] denote the Schouten bracket for multivector fields.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is the jacobiator of P with P in a dictionary format with integer type 'keys' and
            symbol type 'values' if latex format is False or in otherwise is the same result but in latex format

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # Calculates [P, P]
            >>> jacobiator(bivector, latex_format=False)
            >>> {123: 0}
            >>> jacobiator(self, bivector, latex_format=True):
            >>> ''
        """
        return self.lichnerowicz_poisson_operator(bivector, bivector, latex_format=latex_format)

    def is_poisson_bivector(self, bivector):
        """ Calculates de Schouten-Nijenhuis bracket of a given bivector field with himself,
            that is calculates [P,P]_SN.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if [P,P] = 0, in other case is False.

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # [P, P]
            >>> is_poisson_bivector(bivector):
            >>> True
        """
        bivector_result = self.lichnerowicz_poisson_operator(bivector, bivector)
        return True if symbolic_expression(bivector_result, self.dim, self.coordinates).is_zero() else False

    def is_poisson_vector_field(self, bivector, vector_field):
        """ Check if a vector field is a Poisson vector field of a given Poisson bivector field, that is calculate if
            [Z,P] = 0, where Z is vector_field variable and P is bivector variable.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :vector_field:
            Is a vector field in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if [Z,P] = 0, in other case is False.

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For vector field x1*Dx1 + x2*Dx2 + x3*Dx
            >>> vector_field = {1: 'x1', 2: 'x2', 3: 'x3'}
            >>> # [Z, P]
            >>> is_poisson_vector_field(bivector, vector_field)
            >>> False
        """
        bivector_result = self.lichnerowicz_poisson_operator(bivector, vector_field)
        return True if symbolic_expression(bivector_result, self.dim, self.coordinates).is_zero() else False

    def is_poisson_pair(self, bivector_1, bivector_2):
        """ Check if the sum of two Poisson bivector fields P1 and P2 is a Poisson bivector field, that is
            calculate if [P1,P2] = 0

        Parameters
        ==========
        :bivector_1 / bivector_2:
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is True if [P1,P2] = 0, in other case is False.

        Example
        ========
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For vector field x1*Dx1 + x2*Dx2 + x3*Dx
            >>> vector_field = {1: 'x1', 2: 'x2', 3: 'x3'}
            >>> # [Z, P]
            >>> is_poisson_vector_field(bivector, vector_field)
            >>> False
        """
        bivector_result = self.lichnerowicz_poisson_operator(bivector_1, bivector_2)
        return True if symbolic_expression(bivector_result, self.dim, self.coordinates).is_zero() else False

    def modular_vector_field(self, bivector, function, latex_format=False):
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
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :function:
            Is a function scalar h: M --> R that is a non zero and is a string type.

        Returns
        =======
            The result is modular vector field Z of a given Poisson bivector field P relative to the volume form
            in a dictionary format with integer type 'keys' and symbol type 'values' if latex format is False or
            in otherwise is the same result but in latex format

        Example
        =======
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For the function exp:R^3 --> R
            >>> function = 'exp(x1 + x2 + x3)'
            >>> # Calculates Z' = Z - (1/h)*X_h that is (-x2 + x3)*Dx1 + (x1 - x3)*Dx2 + (-x1 + x2)*Dx3
            >>> modular_vector_field(bivector, function)
            >>> {1: -x2 + x3, 2: x1 - x3, 3: -x1 + x2}
            >>> modular_vector_field(bivector, function, latex_format=True)
            >>> ''
        """
        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
            function = is_string(function)
        except Exception as e:
            print(F"[Error modular_vector_field] \n Error: {e}, Type: {type(e)}")

        # Converts strings to symbolic variables
        for key in bivector:
            bivector[key] = sym.sympify(bivector[key])

        # List with 'dim' zeros
        modular_vf_aux = [0] * self.dim
        """ This block calculates the modular vector field Z_0 of a bivector P relative to Omega_0 = dx1^...^dx('dim')
            with the formula: Z0 = -Dxi(Pij)*Dxj + Dxj(Pij)*Dxi, i < j """
        for ij_str in bivector:
            # Convert ij key from str to int
            ij_int = tuple(int(i) for i in str(ij_str))
            modular_vf_aux[ij_int[0] - 1] = sym.simplify(modular_vf_aux[ij_int[0] - 1] + sym.diff(bivector[ij_str],
                                                         self.coordinates[ij_int[1] - 1]))
            modular_vf_aux[ij_int[1] - 1] = sym.simplify(modular_vf_aux[ij_int[1] - 1] - sym.diff(bivector[ij_str],
                                                         self.coordinates[ij_int[0] - 1]))
        modular_vf_0 = dict(zip(range(1, self.dim + 1), modular_vf_aux))

        # Creates a dictionary for the modular vector g(Z) relative to f*Omega_0
        modular_vf_a = {}
        # Formula Z = [(Z0)^i - (1/g)*(X_g)^i]*Dxi
        for z in modular_vf_0:
            modular_vf_a.setdefault(
                z,
                sym.simplify(modular_vf_0[z] - 1/(sym.sympify(function)) * self.hamiltonian_vector_field(bivector,
                                                                                                         function)[z])
            )
        # Return a symbolic type expression or the same expression in latex format
        if latex_format:
            return sym.latex(symbolic_expression(modular_vf_a, self.dim, self.coordinates))
        else:
            return modular_vf_a

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
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> is_homogeneous_unimodular(bivector)
            >>> True
        """
        modular_vector_field = self.self.modular_vector_field(bivector, 1)
        return True if symbolic_expression(modular_vector_field, self.dim, self.coordinates).is_zero() else False

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
            Is a Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :one_form_1/one_form_2:
            Is a one form differential defined on M in a dictionary format with integer type 'keys' and string
            type 'values'.

        Returns
        =======
            The result from calculate {alpha,beta}_P in a dictionary format with integer type 'keys' and symbol
            type 'values' if latex format is False or in otherwise is the same result but in latex format

        Example
        =======
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For one form alpha
            >>> one_form_1 = {1: 'a1*x1', 2: 'a2*x2', 3: 'a3*x3'}
            >>> # For one form beta
            >>> one_form_2 = {1: 'b1*x1', 2: 'b2*x2', 3: 'b3*x3'}
            >>> # Calculates
            >>> # {alpha,beta}_P = (b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2))*(-x2*x3*dx1 - x1*x3*dx2 - x1*x2*dx3)
            >>> one_forms_bracket(bivector, one_form_1, one_form_2)
            >>> {1: -x2*x3*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2)),
                 2: -x1*x3*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2)),
                 3: -x1*x2*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2))}
            >>> modular_vector_field(bivector, one_form_1, one_form_2, latex_format=True)
            >>> ''
        """
        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
            one_form_1 = is_dicctionary(one_form_1)
            one_form_2 = is_dicctionary(one_form_2)
        except Exception as e:
            print(F"[Error one_forms_bracket] \n Error: {e}, Type: {type(e)}")

        # Defines as alpha = one_form_1, beta = one_form_2 and P = bivector.
        # This block converts strings to symbolic variables
        for i in range(1, self.dim + 1):
            one_form_1[i] = sym.sympify(one_form_1[i])
            one_form_2[i] = sym.sympify(one_form_2[i])

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
                    bivector, one_form_1)[z[1]] * (sym.diff(
                        one_form_2[z[0]], self.coordinates[z[1] - 1]) - sym.diff(
                        one_form_2[z[1]], self.coordinates[z[0] - 1])))
            ii_sharp_alpha_d_beta[z[1] - 1] = sym.simplify(
                ii_sharp_alpha_d_beta[z[1] - 1] + self.sharp_morphism(
                    bivector, one_form_1)[z[0]] * (sym.diff(
                        one_form_2[z[1]], self.coordinates[z[0] - 1]) - sym.diff(
                        one_form_2[z[0]], self.coordinates[z[1] - 1])))
            ii_sharp_beta_d_alpha[z[0] - 1] = sym.simplify(
                ii_sharp_beta_d_alpha[z[0] - 1] + self.sharp_morphism(
                    bivector, one_form_2)[z[1]] * (sym.diff(
                        one_form_1[z[0]], self.coordinates[z[1] - 1]) - sym.diff(
                        one_form_1[z[1]], self.coordinates[z[0] - 1])))
            ii_sharp_beta_d_alpha[z[1] - 1] = sym.simplify(
                ii_sharp_beta_d_alpha[z[1] - 1] + self.sharp_morphism(
                    bivector, one_form_2)[z[0]] * (sym.diff(
                        one_form_1[z[1]], self.coordinates[z[0] - 1]) - sym.diff(
                        one_form_1[z[0]], self.coordinates[z[1] - 1])))

        """ This block calculate d_P(alpha,beta) that is the same to d(<beta,P#(alpha)>), where <,> the pairing and
            d(<beta,P#(alpha)>) = d(P#(alpha)^i * beta_i)
        """
        d_pairing_beta_sharp_alpha = sym.simplify(sym.derive_by_array(sum(
            one_form_2[ky] * self.sharp_morphism(bivector, one_form_1)[ky] for ky
            in one_form_2), self.coordinates))

        # List for the coefficients of {alpha,beta}_P,
        one_forms_brack_aux = []
        for i in range(self.dim):
            one_forms_brack_aux.append(sym.simplify(
                ii_sharp_alpha_d_beta[i] - ii_sharp_beta_d_alpha[i]
                + d_pairing_beta_sharp_alpha[i]))

        # Converts one_forms_brack_aux to dictionary
        one_forms_brack = dict(zip(range(1, self.dim + 1),
                                   one_forms_brack_aux))

        # Return a symbolic type expression or the same expression in latex format
        if latex_format:
            return sym.latex(symbolic_expression(one_forms_brack, self.dim, self.coordinates, dx=True))
        else:
            one_forms_brack

    def linear_normal_form_R3(self, bivector):
        """ Calculates a normal form for Lie-Poisson bivector fields on R^3 modulo linear isomorphisms.

        Parameters
        ==========
        :bivector:
            Is a Lie-Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.

        Returns
        =======
            The result is a normal form in a dictionary format with integer type 'keys' and symbol
            type 'values'.

        Example
        =======
            >>> # For bivector x1*Dx2^Dx3
            >>> bivector = {12: 'x1', 13: '0', 23: '0'}
            >>> # Calculates a normal form
            >>> linear_normal_form_R3(bivector)
            >>> {12: 0, 13: 4*a*x2 + x1, 23: 4*a*x1 + x2}
        """
        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
        except Exception as e:
            print(F"[linear_normal_form_R3] \n Error: {e}, Type: {type(e)}")

        # Trivial case
        if all(vl == 0 for vl in bivector.values()):
            return {12: 0, 13: 0, 23: 0}
        # Converts strings to symbolic variables
        for key in bivector:
            bivector[key] = sym.sympify(bivector[key])
        # List for coordinates (x1,x2,x3) on R^3
        x_aux = sym.symbols(f'{self.variable}1:{4}')
        # Remember that E = x1*Dx1 + x2*Dx2 + x3*Dx3 (Euler vector field on R3) and P = bivector
        pairing_E_P = sym.simplify(x_aux[0] * bivector[23] + (-1) * x_aux[1]
                                   * bivector[13] + x_aux[2] * bivector[12])
        # Calculates the Hessian matrix of the pairing <E,P>
        hess_pairing_E_P = sym.simplify(sym.derive_by_array(sym.derive_by_array(
            pairing_E_P, x_aux), x_aux)).tomatrix()
        # Diagonalize the Hessian matrix of the pairing <E,P>
        (trnsf, diag_hess) = hess_pairing_E_P.diagonalize()

        """ Classification:
            Unimodular case. The modular vector field of P relative to the Euclid
            volume form on R^3 is zero.
        """
        if all(vl == 0 for vl in self.modular_vector_field(bivector, 1).values()):
            if hess_pairing_E_P.rank() == 1:
                return {12: 0, 13: 0, 23: x_aux[0]}
            if hess_pairing_E_P.rank() == 2:
                # Case: Hessian of <E,P> with index 2
                if diag_hess[0, 0] * diag_hess[1, 1] > 0 or \
                        diag_hess[0, 0] * diag_hess[2, 2] > 0 or \
                        diag_hess[1, 1] * diag_hess[2, 2] > 0:
                    return {12: 0, 13: -x_aux[1], 23: x_aux[0]}
                else:
                    # Case: Hessian of <E,P> with index 1
                    return {12: 0, 13: x_aux[1], 23: x_aux[0]}
            if hess_pairing_E_P.rank() == 3:
                # Distinguish indices of the Hessian of <E,P>
                index_hess_sum = diag_hess[0, 0]/abs(diag_hess[0, 0]) \
                                 + diag_hess[1, 1]/abs(diag_hess[1, 1]) \
                                 + diag_hess[2, 2]/abs(diag_hess[2, 2])
                if index_hess_sum == 3 or index_hess_sum == -3:
                    return {12: x_aux[2], 13: -x_aux[1], 23: x_aux[0]}
                else:
                    return {12: -x_aux[2], 13: -x_aux[1], 23: x_aux[0]}
        # Non-unimodular case
        else:
            if hess_pairing_E_P.rank() == 0:
                return {12: 0, 13: x_aux[0], 23: x_aux[1]}
            if hess_pairing_E_P.rank() == 2:
                # Case: Hessian of <E,P> with index 2
                if diag_hess[0, 0] * diag_hess[1, 1] > 0 or \
                        diag_hess[0, 0] * diag_hess[2, 2] > 0 or \
                        diag_hess[1, 1] * diag_hess[2, 2] > 0:
                    return {12: 0, 13: x_aux[0] - 4*sym.sympify('a')*x_aux[1],
                            23: x_aux[1] + 4*sym.sympify('a')*x_aux[0]}
                else:
                    # Case: Hessian of <E,P> with index 1
                    return {12: 0, 13: x_aux[0] + 4*sym.sympify('a')*x_aux[1],
                            23: 4*sym.sympify('a')*x_aux[0] + x_aux[1]}
            if hess_pairing_E_P.rank() == 1:
                return {12: 0, 13: x_aux[0], 23: 4*x_aux[0] + x_aux[1]}

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
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector_1 = {12: 'x3', 13: '-x2', 23: 'x1'}
            >>> # For bivector x1*Dx2^Dx3
            >>> bivector_2 = {12: 'x1', 13: '0', 23: '0'}
            >>> is_homogeneous_unimodular(bivector)
            >>> False
        """
        return True if self.linear_normal_form_R3(bivector_1) == self.linear_normal_form_R3(bivector_2) else False

    def gauge_transformation(self, bivector, two_form):
        """"""
        # Validate the params
        try:
            bivector = is_dicctionary(bivector)
            two_form = is_dicctionary(two_form)
        except Exception as e:
            print(F"[gauge_transformation] \n Error: {e}, Type: {type(e)}")

        two_form_B = two_tensor_form_to_matrix(two_form, self.dim)
        I_minus_BP = sym.Matrix(sym.simplify(sym.eye(self.dim) - two_form_B * self.bivector_to_matrix(bivector)))
        det_I_minus_BP = sym.factor(sym.simplify(I_minus_BP.det()))
        if det_I_minus_BP == 0:
            return False
        else:
            gauge_mtx = sym.Matrix(sym.simplify(self.bivector_to_matrix(bivector) * (I_minus_BP.inv())))
        gauge_biv = {}
        for z in itools.combinations(range(1, self.dim + 1), 2):
            gauge_biv.setdefault(int(''.join(str(i) for i in z)), gauge_mtx[z[0]-1, z[1]-1])
        return gauge_biv, sym.sympify(det_I_minus_BP)

    def flaschka_ratiu(self, casimirs, structure_symplectic=False, latex_format=False):
        """"""
        if self.dim - len(casimirs) == 2:
            casimir_list = []
            # Convert all Casimir functions in sympy objects
            for casimir in casimirs:
                casimir_list.append(sym.sympify(casimir))

            # This block calculates d(C), where C is a function Casimir
            diff_casimir_list = []
            diff_casimir_matrix = []
            for casimir in casimir_list:
                casimir_matrix = []
                diff_casimir = 0
                for i, variable in enumerate(self.coordinates):
                    casimir_matrix.append(casimir.diff(variable))
                    diff_casimir = diff_casimir + (casimir.diff(variable) * self.Dx_basis[i])
                diff_casimir_matrix.append(casimir_matrix)
                diff_casimir_list.append(diff_casimir)

            # This block calculates d(C_1)^...^d(C_dimension)
            d_casimir = diff_casimir_list[0]
            for index, element in enumerate(diff_casimir_list):
                d_casimir = d_casimir ^ element if index != 0 else d_casimir

            if d_casimir.is_zero():
                return {}

            # This blocks obtains Poisson coefficients
            bivector_coeff_dict = {}
            combinations = [x for x in itools.combinations(range(1, self.dim + 1), 2)]
            for combination in combinations:
                combination = list(combination)
                casimir_matrix_without_combination = sym.Matrix(diff_casimir_matrix)
                for i, element in enumerate(combination):
                    casimir_matrix_without_combination.col_del(element - (i + 1))

                # Makes a dictionary with Poisson coefficients
                key = int(F'{combination[0]}{combination[1]}')
                coeff = ((-1)**(combination[0] + combination[1]))
                bivector_coeff_dict[key] = sym.simplify(coeff * casimir_matrix_without_combination.det())

            if structure_symplectic:
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
                    bivector = bivector + bivector_coeff_dict[key] * dx_ij_basis[key]
                    estructure_symplectic_num = estructure_symplectic_num + bivector_coeff_dict[key] * dx_ij_basis[key]
                    estructure_symplectic_dem = estructure_symplectic_dem + bivector_coeff_dict[key] * bivector_coeff_dict[key] # noqa

                symplectic = estructure_symplectic_num * (-1 / estructure_symplectic_dem)

                if latex_format:
                    return [sym.latex(bivector),
                            sym.latex(symplectic)]
                else:
                    return [bivector, symplectic]
            else:
                return sym.latex(symbolic_expression(bivector_coeff_dict, self.dim, self.coordinates)) if latex_format else bivector_coeff_dict # noqa
        else:
            return {}
