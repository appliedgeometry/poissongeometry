# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sympy as sym
import itertools as itools
from utils import (validate_dimension, symbolic_expression,
                   is_dicctionary, show_coordinates,
                   two_tensor_form_to_matrix, bivector_to_matrix)


class PoissonGeometry:
    """ This class provides some useful tools for Poisson-Nijenhuis calculus on Poisson manifolds."""

    def __init__(self, dimension, variable='x'):
        # Obtains the dimension
        self.dim = validate_dimension(dimension)
        # Define what variables the class will work with
        self.variable = variable
        # Create the symbolics symbols
        self.coordinates = sym.symbols(f'{self.variable}1:{self.dim + 1}')
        # Show the coordinates with that will the class works
        self.coords = show_coordinates(self.coordinates)

    def sharp_morphism(self, bivector, one_form, latex_format=False):
        """ Calculates the image of a differential 1-form under the vector bundle morphism 'sharp' P#: T*M -> TM
        defined by P#(alpha) := i_(alpha)P, where P is a Poisson bivector field on a manifold M, alpha a 1-form on M
        and i the interior product of alpha and P.

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
            >>> print(sharp_morphism(bivector, one_form, latex_format=True))
            >>> 'x_{2} x_{3} \\left(- a_{2} + a_{3}\\right) \\boldsymbol{Dx}_{1} + x_{1} x_{3} \\left(a_{1}
            - a_{3}\\right) \\boldsymbol{Dx}_{2} + x_{1} x_{2} \\left(- a_{1} + a_{2}\\right) \\boldsymbol{Dx}_{3}'
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
            >>> one_form = {1: 'x1', 2: 'x2', 3: 'x3'}.
            >>> # P#(one_form) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
            >>> is_in_kernel(bivector, one_form)
            >>> True
        """
        p_sharp = self.sharp_morphism(bivector, one_form)
        # Converts a dictionary symbolic to a symbolic expression and verify is zero with a sympy method
        return True if symbolic_expression(p_sharp, self.dim, self.coordinates, self.variable).is_zero() else False

    def hamiltonian_vector_field(self, bivector, hamiltonian_function, latex_syntax=False):
        """ Calculates the Hamiltonian vector field of a function rela-
        tive to a (Poisson) bivector field.
        Remark:
        The Hamiltonian vector field of a scalar function h on a Poisson manifold (M,P),
        relative to the Poisson bivector field P, is given by
            X_h = P#(dh),
        where dh is the exterior derivative (differential) of h and P#: T*M -> TM is the
        vector bundle morphism defined by
            P#(alpha) := i_(alpha)P,
        with i is the interior product of alpha and P.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param hamiltonian_function: is a string type variable. For example, on
         R^3,
            'x1 + x2 + x3'.
        :return: a dictionary with integer type 'keys' and symbol type
         'values'. For the previous example,
            {1: x2 - x3, 2: -x1 + x3, 3: x1 - x2},
         which represents the Hamiltonian vector field
            X_h = (x2 - x3)*Dx1 + (-x1 + x3)*Dx2 + (x1 - x2)*Dx3.
        """
        # Calculates the differential of hamiltonian_function
        dh = sym.derive_by_array(sym.sympify(hamiltonian_function), self.coordinates)
        # Calculates the Hamiltonian vector field
        ham_vector_field = self.sharp_morphism(bivector, dict(zip(range(1, self.dim + 1), dh)))

        # return a dictionary
        return sym.latex(dict_to_symbol_exp(ham_vector_field, self.dim ,self.coordinates)) if latex_syntax else ham_vector_field


    def is_casimir(self, bivector, function):
        """ Check if a function is a Casimir function of a given (Poisson) bivector field.
        Remark:
        A function K on a Poisson manifold (M,P) is said to be a Casimir function of the
        Poisson bivector field P if
            P#(dK) = 0,
        where dK is the exterior derivative (differential) of K and P#: T*M -> TM is the
        vector bundle morphism defined by
            P#(alpha) := i_(alpha)P,
        with i the interior product of alpha and P.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param function: is a string type variable. For example, on R^3,
            'x1**2 + x2**2 + x3**2'.
        :return: a boolean type variable.
        """
        # Check if the Hamiltonian vector field of K is zero or not
        if all(val == 0 for val in self.hamiltonian_vector_field(bivector, function).values()):
            return True
        return False


    def poisson_bracket(self, bivector, function_1, function_2, latex_syntax=False):
        """ Calculates the poisson bracket of two functions induced by
        a given Poisson bivector field.
        Remark:
        The Poisson bracket of two functions f and g on a Poisson manifold (M,P) is given by
            {f,g} = π(df,dg) = ⟨dg,π#(df)⟩,
        where df and dg are the exterior derivatives (differentials) of f and g respectively,
        and P#: T*M -> TM is the vector bundle morphism defined by
            P#(alpha) := i_(alpha)P,
        with i the interior product of alpha and P. Also ⟨,⟩ denote the natural pairing between
        differential forms and vector fields.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param function_1: is a string type variable. For example, on
         R^3,
            'x1 + x2 + x3'.
        :param function_2: is a string type variable. For example, on
         R^3
            'a1*x1 + a2*x2 + a3*x3'.
        :return: a symbolic type expression. For the previous example,
            a1*(x2 - x3) - a2*(x1 - x3) + a3*(x1 - x2)
        """
        f1, f2 = sym.sympify(function_1), sym.sympify(function_2)
        if f1 - f2 == 0:
            # {f1,f2} = 0 for f1=f2
            return 0
        # Calculates the differentials of function_1 and function_2
        df1 = dict(zip(range(1, self.dim + 1), sym.derive_by_array(f1, self.coordinates)))
        df2 = dict(zip(range(1, self.dim + 1), sym.derive_by_array(f2, self.coordinates)))
        # Calculates {f1,f2} = <df2,P#(df1)> = ∑_{i} (P#(df1))^i * (df2)_i
        bracket_f1_f2 = sym.simplify(sum(self.sharp_morphism(bivector, df1)[index] * df2[index] for index in df1))

        # Return a symbolic type expression
        return sym.latex(bracket_f1_f2) if latex_syntax else bracket_f1_f2


    def lichnerowicz_poisson_operator(self, bivector, multivector, latex_syntax=False):
        """ Calculates the Schouten-Nijenhuis bracket between a given
        (Poisson) bivector field and a (arbitrary) multivector field.
        Recall that the Lichnerowicz-Poisson operator on a Poisson ma-
        nifold (M,P) is defined as the adjoint operator of P, ad_P,
        respect to the Schouten bracket for multivector fields on M:
            ad_P(A) := [P,A],
        for any multivector field A on M. Here [,] denote the Schouten-
        Nijenhuis bracket.
        Let P = Pij*Dxi^Dxj (i < j) and A = A^J*Dxj_1^Dxj_2^...^Dxj_a.
        Here, we use the multi-index notation A^J := A^j_1j_2...j_a,
        for j_1 < j_2 <...< j_a. Then,
        [P,A](df1,...,df(a+1))
                =
        sum_(i=1)^(a+1) (-1)**(i)*{fi,A(df1,...,î,...,df(a+1))}_P
        + sum(1<=i<j<=a+1) (-1)**(i+j)*A(d{fi,fj}_P,..î..^j..,df(a+1)),
        for any smooth functions fi on M. Where {,}_P is the Poisson
        bracket induced by P. Here ^ denotes the absence of the corres-
        ponding index and dfi is the differential of fi.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param multivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {123: 'x1*x2*x3'}.
         The 'keys' are the ordered indices of a given multivector field
         A and the 'values' their coefficients. In this case,
            A = x1*x2*x3*Dx1^Dx2*Dx3.
        :return: a dictionary with integer type 'keys' and symbol type
         'values' or the zero value. For the previous example,
            0,
         which says that any 4-multivector field on R^3 is zero.
        """
        mltv = multivector
        deg_mltv = len(tuple(str(next(iter(mltv)))))  # Degree of multivector
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
                        lich_poiss_aux_1 = sym.simplify(lich_poiss_aux_1
                                                       + lich_poiss_aux_11)
                    # Add the brackets {xi,A^J} in the list schouten_biv_mltv_1
                    schouten_biv_mltv_1.setdefault(
                        int(''.join(str(j) for j in z)), lich_poiss_aux_1)
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
                            nw_idx_mltv.append(i)  # Add the index i
                            nw_idx_mltv.sort()  # Sort the indices
                            if nw_idx_mltv.count(i) != 2:  # Ignore repet. indx
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
                    # Add the terms Dxk(P^iris)*A^((J+k)\iris)) in the list
                    # schouten_biv_mltv_2
                    schouten_biv_mltv_2.setdefault(
                        int(''.join(str(j) for j in z)), sum_y)
                    sum_y = 0
                    sum_y_aux = 0
                # A dictionary for the the Schouten bracket [P,A]
                schouten_biv_mltv = {}
                # Sum and add the terms in
                # schouten_biv_mltv_1 and schouten_biv_mltv_2
                for ky in schouten_biv_mltv_1:
                    schouten_biv_mltv.setdefault(
                        ky, sym.simplify(schouten_biv_mltv_1[ky]
                                        + schouten_biv_mltv_2[ky]))
                # return a dictionary
                return sym.latex(dict_to_symbol_exp(schouten_biv_mltv, self.dim ,self.coordinates)) if latex_syntax else schouten_biv_mltv


    def jacobiator(self, bivector, latex_syntax=False):
        """ Calculates de Schouten-Nijenhuis bracket of a given bivec-
        tor field with himself.
        
        Remark:
        Given a bivector field P on a smooth manifold M, the Jacobiator
        of P is defined by
            Jacobiator(P) = [P,P],
        where [,] denote the Schouten bracket for multivector fields.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :return: a dictionary with integer type 'keys' and symbol type
         'values'. For the previous example,
            {123: 0}
        """
        return self.lichnerowicz_poisson_operator(bivector, bivector, latex_syntax=latex_syntax)


    def is_poisson(self, bivector):
        """ Calculates de Schouten-Nijenhuis bracket of a given bivec-
        tor field with himself.
        Remark:
        A bivector field P is said to be a Poisson tensor,
        or Poisson bivector field, if (and only if)
            [P,P] = 0,
        where [,] denote the Schouten bracket for multivector fields.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :return: a string type variable. For the previous example,
            'Is a Poisson tensor.'
        """
        if all(value == 0 for value in self.lichnerowicz_poisson_operator(bivector, bivector).values()):
            return True
        return False


    def is_poisson_vector_field(self, bivector, vector_field):
        """ Check if a vector field is a Poisson vector field of a
        given Poisson bivector field.
        Remark:
        A vector field Z on a Poisson manifold (M,P) is said to be a
        Poisson vector field if (and only if)
            [Z,P] = 0,
        where [,] denote the Schouten bracket for multivector fields.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param vector_field: is a dictionary with integer type 'keys'
        and string type 'values'. For example, on R^3,
            {1: 'x1', 2: 'x2', 3: 'x3'}.
         The 'keys' are the indices of the coefficients of a given vec-
         tor field Z and the 'values' their coefficients. In this case,
            Z = x1*Dx1 + x2*Dx2 + x3*Dx3.
        :return: a string type variable. For the previous example,
            'Is not a Poisson vector field of the Poisson tensor.'
        """
        if all(value == 0 for value in self.lichnerowicz_poisson_operator(bivector, vector_field).values()):
            return True
        return False


    def is_poisson_pair(self, bivector_1, bivector_2):
        """ Check if the sum of two Poisson bivector fields is a
        Poisson bivector field.
        Recall that two Poisson bivector fields P1 and P2 is said to be
        a Poisson pair if the sum P1 + P2 is again a Poisson bivector
        field. Equivalently, if
            [P1,P2] = 0,
        where [,] denote the Schouten bracket for multivector fields.
        :param bivector_1: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param bivector_2: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3**2', 13: '-x2**2', 23: 'x1**2'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P2 = x3**2*Dx1^Dx2 - x2**2*Dx1^Dx3 + x1**2*Dx2^Dx3.
        :return: a string type variable. For the previous example,
            'Is a Poisson pair.'
        """
        if all(value == 0 for value in self.lichnerowicz_poisson_operator(bivector_1, bivector_2).values()):
            return True
        return False


    def modular_vector_field(self, bivector, function, latex_syntax=False):
        """ Calculates the modular vector field of a given Poisson bi-
        vector field relative to the volume form
            ('function')*Omega_0,
        with Omega_0 = dx1^...^dx('dim'). We assume that 'function' is
        nowhere vanishing.
        Recall that, on a orientable Poisson manifold (M,P,Omega), the
        modular vector field Z of the Poisson bivector field P, relati-
        ve to the volume form Omega, is defined by
            Z(f) := div(X_f),
        for all function f on M. Here, div(X_f) is the divergence res-
        pect to Omega of the Hamiltonian vector field X_f of f relati-
        ve to P. Clearly, the modular vector field is Omega-dependent.
        If g*Omega is another volume form, with g a nowhere vanishing
        function on M, then
            Z' = Z - (1/g)*X_g,
        is the modular vector field of P relative to g*Omega.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param function: is a string type variable. For example, on
         R^3,
            'exp(x1 + x2 + x3)'.
        :return: a dictionary with integer type 'keys' and symbol type
         'values'. For the previous example,
            {1: -x2 + x3, 2: x1 - x3, 3: -x1 + x2},
         which represents the Modular vector field
            Z = (-x2 + x3)*Dx1 + (x1 - x3)*Dx2 + (-x1 + x2)*Dx3.
        """
        # Converts strings to symbolic variables
        for key in bivector:
            bivector[key] = sym.sympify(bivector[key])
        # List with 'dim' zeros
        modular_vf_aux = [0] * self.dim
        # Calculates the modular vector field Z0 of 'bivector' of
        # P == 'bivector' relative to Omega_0 = dx1^...^dx('dim')
        # Formula: Z0 = -Dxi(Pij)*Dxj + Dxj(Pij)*Dxi, i < j
        for ij_str in bivector:
            # Convert ij key from str to int
            ij_int = tuple(int(i) for i in str(ij_str))
            modular_vf_aux[ij_int[0] - 1] = sym.simplify(modular_vf_aux[ij_int[0] - 1] +
                                                        sym.diff(bivector[ij_str],
                                                                self.coordinates[ij_int[1] - 1]))
            modular_vf_aux[ij_int[1] - 1] = sym.simplify(modular_vf_aux[ij_int[1] - 1] -
                                                        sym.diff(bivector[ij_str],
                                                                self.coordinates[ij_int[0] - 1]))

        modular_vf_0 = dict(zip(range(1, self.dim + 1), modular_vf_aux))
        # Dictionary for the modular vector f. Z relative to g*Omega_0
        # g == 'function'
        modular_vf_a = {}
        # Formula Z = [(Z0)^i - (1/g)*(X_g)^i]*Dxi
        for z in modular_vf_0:
            modular_vf_a.setdefault(z, sym.simplify(modular_vf_0[z] - 1/(sym.sympify(function)) * self.hamiltonian_vector_field(bivector, function)[z]))
        # Return a dictionary
        return sym.latex(dict_to_symbol_exp(modular_vf_a, self.dim ,self.coordinates)) if latex_syntax else modular_vf_a


    def is_homogeneous_unimodular(self, bivector):
        """ Check if a homogeneous Poisson bivector field is unimodular
        or not.
        Recall that a homogeneous Poisson bivector field on R^m is uni-
        modular on (all) R^m if and only if their modular vector field
        relative to the Euclidean volume form is zero.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :return: a string type variable. For the previous example,
            'The homogeneous Poisson tensor is unimodular on R^3.'
        """
        if all(value == 0 for value in self.modular_vector_field(bivector, 1).values()):
            return True
        return False


    def one_forms_bracket(self, bivector, one_form_1, one_form_2, latex_syntax=False):
        """ Calculates the Lie bracket of two differential 1-forms in-
        duced by a given Poisson bivector field.
        Recall that the Lie bracket of two differential 1-forms alpha
        and beta on a Poisson manifold (M,P) is defined by
            {alpha,beta}_P := i_P#(alpha)(d_beta) - i_P#(beta)(d_alpha)
                            + d_P(alpha,beta),
        where d_alpha and d_beta are the exterior derivative of alpha
        and beta, respectively, i_ the interior product of vector fields
        on differential forms, P#: T*M -> TM the vector bundle morphism
        defined by P#(alpha) := i_(alpha)P, with i the interior product
        of alpha and P. Note that, by definition,
            {df,dg}_P = d_{f,g}_P,
        for ant functions f,g on M.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param one_form_1: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {1: 'a1*x1', 2: 'a2*x2', 3: 'a3*x3'}.
         The 'keys' are the indices of the coefficients of a given
         differential 1-form alpha and the 'values' their coefficients.
         In this case,
            alpha = a1*x1*dx1 + a2*x2*dx2 + a3*x3*dx3.
        :param one_form_2: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {1: 'b1*x1', 2: 'b2*x2', 3: 'b3*x3'}.
         The 'keys' are the indices of the coefficients of a given
         differential 1-form beta and the 'values' their coefficients.
         In this case,
            beta = b1*x1*dx1 + b2*x2*dx2 + b3*x3*dx3.
        :return: a dictionary with integer type 'keys' and
         string type 'values'. For the previous example,
         {1: -x2*x3*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2)),
          2: -x1*x3*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2)),
          3: -x1*x2*(b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2))},
         which represents the differential 1-form
            c*(-x2*x3*dx1 - x1*x3*dx2 - x1*x2*dx3),
         with c = (b1*(a2 - a3) - b2*(a1 - a3) + b3*(a1 - a2)).
        """
        for i in range(1, self.dim + 1):
            # Converts strings to symbolic variables
            one_form_1[i] = sym.sympify(one_form_1[i])
            one_form_2[i] = sym.sympify(one_form_2[i])
        # List with 'dim' zeros
        ii_sharp_alpha_d_beta = [0] * self.dim  # alpha == one_form_1
        ii_sharp_beta_d_alpha = [0] * self.dim  # beta == one_form_2
        # Calculates the terms first and second terms of the Lie brcket
        # i_P#(alpha)(d_beta) and i_P#(beta)(d_alpha)
        # Formula: i_P#(alpha)(d_beta) =
        # Xj*(Dxj(bta_i)-Dxi(bta_j))*dxi + Xi*(Dxi(bta_j)-Dxj(bta_i))dxj
        # for i<j. Here, X==P#(alpha). Analogous for i_P#(beta)(d_alpha)
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
        # Calculates the third term of the Lie bracket
        # d_P(alpha,beta) = d(<beta,P#(alpha)>), with <,> the pairing
        # Formula: d(<beta,P#(alpha)>) = d(P#(alpha)^i * beta_i)
        d_pairing_beta_sharp_alpha = sym.simplify(sym.derive_by_array(sum(
            one_form_2[ky] * self.sharp_morphism(bivector, one_form_1)[ky] for ky
            in one_form_2), self.coordinates))
        # List for the coefficients of {alpha,beta}_P, P == 'bivector'
        one_forms_brack_aux = []
        for i in range(self.dim):
            one_forms_brack_aux.append(sym.simplify(
                ii_sharp_alpha_d_beta[i] - ii_sharp_beta_d_alpha[i]
                + d_pairing_beta_sharp_alpha[i]))
        # Converts one_forms_brack_aux to dictionary
        one_forms_brack = dict(zip(range(1, self.dim + 1),
                                   one_forms_brack_aux))
        return sym.latex(dict_to_symbol_exp(one_forms_brack, self.dim ,self.coordinates, dx=True)) if latex_syntax else one_forms_brack  # Return a dictionary


    def linear_normal_form_R3(self, bivector):
        """ Calculates a normal form for Lie-Poisson bivector fields on
        R^3 modulo linear isomorphisms.
        Recall that two Lie-Poisson bivector fields P1 and P2 on R^3 is
        said to be equivalents if there exists a linear isomorphism T:
        R^3 -> R^3 such that T*P2 = P1.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x1', 13: '0', 23: '0'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x1*Dx1^Dx2.
        :return: a dictionary with integer type 'keys' and
         string type 'values'. For the previous example,
            {12: 0, 13: 4*a*x2 + x1, 23: 4*a*x1 + x2},
         which represents the normal form
            L = (4*a*x2 + x1)*Dx1^Dx3 + (4*a*x1 + x2)*Dx2^Dx3,
         for some a > 0.
        """
        # Trivial case
        if all(vl == 0 for vl in bivector.values()):
            return {12: 0, 13: 0, 23: 0}
        # Converts strings to symbolic variables
        for key in bivector:
            bivector[key] = sym.sympify(bivector[key])
        # List for coordinates (x1,x2,x3) on R^3
        x_aux = sym.symbols(f'{self.variable}1:{4}')
        # Here, E is the Euler vector field on R3, E = x1*Dx1 + x2*Dx2 + x3*Dx3
        # P == 'bivector'
        pairing_E_P = sym.simplify(x_aux[0] * bivector[23] + (-1) * x_aux[1]
                                  * bivector[13] + x_aux[2] * bivector[12])
        # Calculates the Hessian matrix of the pairing <E,P>
        hess_pairing_E_P = sym.simplify(sym.derive_by_array(sym.derive_by_array(
            pairing_E_P, x_aux), x_aux)).tomatrix()
        # Diagonalize the Hessian matrix of the pairing <E,P>
        (trnsf, diag_hess) = hess_pairing_E_P.diagonalize()
        # Classification:
        # Unimodular case. The modular vector field of P relative to the Euclid
        # volume form on R^3 is zero
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
        """ Determines if two Lie-Poisson bivector fields on R^3 are
        isomorphic or not.
        Remark:
        Two Lie-Poisson bivector fields P1 and P2 on R^3 are said to be
        equivalents, or isomorphics, if there exists a linear isomorphism
        T: R^3 -> R^3 such that T*P2 = P1
        :param bivector_1: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.
        :param bivector_2: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x1', 13: '0', 23: '0'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x1*Dx1^Dx2.
        :return: a string type variable. For the previous example,
            'The Lie-Poisson tensors are not isomorphic.'
        """
        if self.linear_normal_form_R3(bivector_1) == self.linear_normal_form_R3(bivector_2):
            return True
        return False


    def gauge_transformation(self, bivector, two_form):
        two_form_B = sym.MatrixSymbol('B', self.dim, self.dim)
        two_form_B = sym.Matrix(two_form_B)
        # Assigns the corresponding coefficients of the 2-form
        for z in itools.combinations_with_replacement(range(1, self.dim + 1),
                                                      2):
            if z[0] == z[1]:
                two_form_B[z[0] - 1, z[1] - 1] = 0
            else:
                two_form_B[z[0] - 1, z[1] - 1] = sym.sympify(
                    two_form[int(''.join(str(i) for i in z))])
                two_form_B[z[1] - 1, z[0] - 1] = (-1) * two_form_B[
                    z[0] - 1, z[1] - 1]
        I_minus_BP = sym.Matrix(sym.simplify(
            sym.eye(self.dim) - two_form_B * self.bivector_to_matrix(bivector)))
        det_I_minus_BP = sym.factor(sym.simplify(I_minus_BP.det()))
        if det_I_minus_BP == 0:
            return False
        else:
            gauge_mtx = sym.Matrix(
                sym.simplify(self.bivector_to_matrix(bivector) * (I_minus_BP.inv())))
        gauge_biv = {}
        for z in itools.combinations(range(1, self.dim + 1),
                                                      2):
            gauge_biv.setdefault(int(''.join(str(i) for i in z)),
                                 gauge_mtx[z[0]-1, z[1]-1])
        return gauge_biv, sym.sympify(det_I_minus_BP)


    def flaschka_ratiu(self, bivector, casimirs, structure_symplectic=False, latex_syntax=False):
        ''''''
        for key in bivector.keys():
            bivector[key] = sym.sympify(bivector[key])
        if self.dim - len(casimirs) == 2:
            casimir_list = []
            for casimir in casimirs:
                casimir_list.append(sym.sympify(casimir))

            # This block calculates d(C), where C is a function Casimir
            diff_casimir_list = []
            diff_casimir_matrix = []
            for casimir in casimir_list:
                casimir_matrix = []
                diff_casimir = 0
                for i, variable in enumerate(variables_list):
                    casimir_matrix.append(casimir.diff(variable))
                    diff_casimir = diff_casimir + (casimir.diff(variable) * dx_list[i])
                diff_casimir_matrix.append(casimir_matrix)
                diff_casimir_list.append(diff_casimir)

            # This block calculates d(C_1)^...^d(C_dimension)
            d_casimir = diff_casimir_list[0]
            for index, element in enumerate(diff_casimir_list):
                d_casimir = d_casimir ^ element if index != 0 else d_casimir
    
            if d_casimir.is_zero():
                return f'Tu bivector es cero :('

            # This blocks obtains Poisson coefficients
            bivector_coeff_dict = {}
            combinations = [x for x in itertools.combinations(range(1, dimension + 1), 2)]
            for combination in combinations:
                combination = list(combination)
                casimir_matrix_without_combination = sympy.Matrix(diff_casimir_matrix)
                for i, element in enumerate(combination):
                    casimir_matrix_without_combination.col_del(element - (i + 1))

                # Makes a dictionary with Poisson coefficients
                key = int(F'{combination[0]}{combination[1]}')
                coeff = ((-1)**(combination[0] + combination[1]))
                bivector_coeff_dict[key] = sympy.simplify(coeff * casimir_matrix_without_combination.det())

            if structure_symplectic:
                variables = variable_string(dim, symbol='Dx')
                basis = Ga(variables, g=sym.eye(dim), coords=coordinates)
                basis = basis.mv()

                # Makes a basis {dxi^dxj}_{i<j} and it save in a dictionary type variable
                dx_ij_basis = {}
                i = 0
                while i < len(dx_list):
                    j = i + 1
                    while j < len(dx_list):
                        key = int(F"{i+1}{j+1}")
                        dx_ij_basis[key] = dx_list[i] ^ dx_list[j]
                        j = j + 1
                    i = i + 1

                bivector = 0
                estructure_symplectic_num = 0
                estructure_symplectic_dem = 0
                for key in dx_ij_basis.keys():
                    bivector = bivector + bivector_coeff_dict[key] * dx_ij_basis[key]
                    estructure_symplectic_num = estructure_symplectic_num + bivector_coeff_dict[key] * dx_ij_basis[key]
                    estructure_symplectic_dem = estructure_symplectic_dem + bivector_coeff_dict[key] * bivector_coeff_dict[key]

                symplectic = estructure_symplectic_num * (-1 / estructure_symplectic_dem)

                if latex_syntax:
                    return [sym.latex(dict_to_symbol_exp(bivector, self.dim ,self.coordinates)),
                            sym.latex(dict_to_symbol_exp(symplectic, self.dim ,self.coordinates))]
                else:
                    return [bivector, symplectic]
            else:
                return sym.latex(dict_to_symbol_exp(bivector_coeff_dict, self.dim ,self.coordinates)) if latex_syntax else bivector_coeff_dict
        else:
            return F'No es posible aplicar flaska ratiu'