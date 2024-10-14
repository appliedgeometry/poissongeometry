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
import collections
import itertools as itls
from poisson.errors import (MultivectorError, FunctionError,
                            DiferentialFormError, CasimirError,
                            DimensionError, Nonlinear,
                            Nonhomogeneous)
from poisson.utils import validate_dimension, del_columns, show_coordinates


class PoissonGeometry:
    """ This class provides some useful tools for Poisson-Nijenhuis calculus on Poisson manifolds."""

    def __init__(self, dimension, variable='x'):
        # Obtains the dimension
        self.dim = validate_dimension(dimension)
        # Define what variables the class will work with
        self.variable = variable
        # Create the symbolics symbols
        self.coords = sym.symbols(f'{self.variable}1:{self.dim + 1}')
        # Print the coordinates
        self.coordinates = show_coordinates(self.coords)

    def bivector_to_matrix(self, bivector, latex=False):
        """ Constructs the matrix of a 2-contravariant tensor field or bivector field.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :latex:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False

        Returns
        =======
            The result is a symbolic skew-symmetric matrix of dimension (dim)x(dim) if latex is False
            in otherwise the result is the same but in latex format.

        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> pg.bivector_to_matrix(bivector, latex=False)
            >>> Matrix([[0, x3, -x2], [-x3, 0, x1], [x2, -x1, 0]])
            >>> pg.bivector_to_matrix(bivector, latex=True)
            >>> '\\left[\\begin{matrix}0 & x_{3} & - x_{2}\\\\- x_{3} & 0 & x_{1}\\\\x_{2} &
                - x_{1} & 0\\end{matrix}\\right]'
        """
        bivector_matrix = sym.zeros(self.dim + 1)

        # Assigns the corresponding coefficients of the bivector field
        for e in bivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F"repeated indexes {e} in {bivector}")
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F"invalid key {e} in {bivector}")
            # Makes the Poisson Matrix
            bivector_matrix[e] = bivector[e]
            # Get the index in the opposite direction
            swap_e = e[::-1]
            bivector_matrix[swap_e] = (-1) * bivector_matrix[e]

        # Return a symbolic Poisson matrix or the same expression in latex format
        return sym.latex(bivector_matrix) if latex else bivector_matrix[1:, 1:]

    def sharp_morphism(self, bivector, one_form, latex=False):
        """ Calculates the image of a differential 1-form under the vector bundle morphism 'sharp' P#: T*M -> TM
            defined by P#(alpha) := i_(alpha)P, where P is a Poisson bivector field on a manifold M, alpha a 1-form
            on M and i the interior product of alpha and P.
        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :one_form:
            Is a 1-form differential in a dictionary format with tuple type 'keys' and string type 'values'.
        :latex:
            Is a boolean flag to indicates if the result is given in latex format or not, its default value is False
        Returns
        =======
            The result is the image of a differential 1-form alpha under the vector bundle morphism 'sharp' in a
            dictionary format with tuple type 'keys' and symbol type 'values'
        Example
        ========
            >>> # Instance the class to dimension 3
            >>> pg = PoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # For one form a1*x1*dx1 + a2*x2*dx2 + a3*x3*dx3.
            >>> one_form = {(1,): 'a1*x1', (2,): 'a2*x2', (3,): 'a3*x3'}
            >>> # P#(one_form) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
            >>> pg.sharp_morphism(bivector, one_form, latex=False)
            >>> {(1,): x2*(-a2*x3 + a3*x3), (2,): x1*(a1*x3 - a3*x3), (3,): x1*x2*(-a1 + a2)}
            >>> sharp_morphism(bivector, one_form, latex=True)
            >>> '\\left\\{ \\left( 1,\\right) : - a_{2} x_{2} x_{3} + a_{3} x_{2} x_{3},
                 \\ \\left( 2,\\right) : a_{1} x_{1} x_{3} - a_{3} x_{1} x_{3},
                 \\ \\left( 3,\\right) : - a_{1} x_{1} x_{2} + a_{2} x_{1} x_{2}\\right\\}'
        """
        one_form_vector = sym.zeros(self.dim + 1, 1)
        for e in one_form:
            if e[0] <= 0:
                raise DiferentialFormError(F"invalid key {e} in {one_form}")
            one_form_vector[int(*e)] = one_form[e]

        one_form_vector = one_form_vector[1:, :]
        bivector_matrix = self.bivector_to_matrix(bivector)

        if not isinstance(bivector_matrix, sym.matrices.dense.MutableDenseMatrix):
            return bivector_matrix

        """
            Calculation of the vector field bivector^#(one_form) as
                ∑_{i}(∑_{j} bivector_ij (one_form_i*Dx_j - one_form_j*Dx_i)
            where i < j.
        """
        sharp_vector = (-1) * (bivector_matrix * one_form_vector)
        # Return a vector field expression in LaTeX format or a symbolic dictionary.
        if latex:
            sharp = {(e + 1,): sym.simplify(sharp_vector[e]) for e in range(self.dim)}
            return sym.latex(sharp)
        else:
            return {(e + 1,): f"{sym.simplify(sharp_vector[e])}" for e in range(self.dim)}

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
            >>> one_form = {(1,): 'x1', (2,): 'x2', (3,): 'x3'}
            >>> # P#(one_form) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
            >>> pg.is_in_kernel(bivector, one_form)
            >>> True
        """
        image = self.sharp_morphism(bivector, one_form)
        if not isinstance(image, dict):
            return image
        if all(value == '0' for value in image.values()):
            image = {}
        return False if bool(image) else True

    def hamiltonian_vf(self, bivector, ham_function, latex=False):
        """ Calculates the Hamiltonian vector field of a function relative to a Poisson bivector field as follows:
            X_h = P#(dh), where d is the exterior derivative of h and P#: T*M -> TM is the vector bundle morphism
            defined by P#(alpha) := i_(alpha)P, with i is the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :hamiltonian_function:
            Is a function scalar h: M --> R that is a string type.
        :latex:
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
            >>> pg.hamiltonian_vf(bivector, hamiltonian_function, latex=False)
            >>> {(1,): x2 - x3, (2,): -x1 + x3, (3,): x1 - x2}
            >>> pg.hamiltonian_vf(bivector, hamiltonian_function, latex=True)
            >>> \\left ( x_{2} - x_{3}\\right ) \\boldsymbol{Dx}_{1} + \\left ( - x_{1} + x_{3}\\right )
                \\boldsymbol{Dx}_{2} + \\left ( x_{1} - x_{2}\\right ) \\boldsymbol{Dx}_{3}'
        """
        hh = sym.sympify(ham_function)
        # Calculates the differential of hamiltonian_function
        d_hh = sym.derive_by_array(hh, self.coords)
        d_hh = {(e + 1,): d_hh[e] for e in range(self.dim)}
        # Calculate and return the Hamiltonian vector field
        return self.sharp_morphism(bivector, d_hh, latex=latex)

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
        ham_vf = self.hamiltonian_vf(bivector, function)
        if not isinstance(ham_vf, dict):
            return ham_vf
        if all(value == '0' for value in ham_vf.values()):
            ham_vf = {}
        return False if bool(ham_vf) else True

    def poisson_bracket(self, bivector, function_1, function_2, latex=False):
        """ Calculates the poisson bracket {f,g} = π(df,dg) = ⟨dg,π#(df)⟩ of two functions
            f and g on a Poisson manifold (M,P), where d is the exterior derivatives and P#: T*M -> TM is the
            vector bundle morphism defined by P#(alpha) := i_(alpha)P, with i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :f / g:
            Is a function scalar f/g: M --> R that is a string type.
        :latex:
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
            >>> pg.poisson_bracket(bivector, f, g, latex=False)
            >>> a1*(x2 - x3) - a2*(x1 - x3) + a3*(x1 - x2)
            >>> pg.poisson_bracket(bivector, f, g, latex=True)
            >>> 'a_{1} \\left(x_{2} - x_{3}\\right) - a_{2} \\left(x_{1} - x_{3}\\right) + a_{3} \\left(x_{1}
                - x_{2}\\right)'
        """
        # Convert to symbolic the expression function_1-function_2
        ff_min_gg = sym.sympify(f'({function_1}) - ({function_2})')
        ff_min_gg = sym.simplify(ff_min_gg)
        if ff_min_gg == 0:
            # That signifies that if ffunction_1=function_2 then {function_1,function_2} = 0
            return sym.sympify(0)

        for f in [function_1, function_2]:
            if self.is_casimir(bivector, f):
                # That signifies that {function_1, function_2} = 0
                return sym.sympify(0)

        # Calculates the differential of function_2
        gg = sym.sympify(function_2)
        d_gg = sym.derive_by_array(gg, self.coords)
        ff_ham_vf = self.hamiltonian_vf(bivector, function_1)

        if not isinstance(ff_ham_vf, dict):
            return ff_ham_vf

        # Calculates {f,g}
        bracket = [d_gg[e[0] - 1] * sym.sympify(ff_ham_vf[e]) for e in ff_ham_vf]
        bracket = sum(bracket)

        # Return a vector field expression in LaTeX format or a symbolic dictionary.
        return sym.latex(bracket) if latex else f"{bracket}"

    def curl_operator(self, multivector, function, latex=False):
        """ Calculate the divergence of multivertoc field.

        Parameters
        ==========
        :multivector:
            Is a multivector filed in a dictionary format with integer type 'keys' and string type 'values'.
        :function:
            Is a nowhere vanishing function in a string type. If the function is constant you can input the
            number type.
        :latex:
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
            >>> pg.curl operator(multivector, function,latex=False)
            >>> '0'
        """
        if sym.simplify(sym.sympify(function)) == 0:
            raise FunctionError(F'Fuction {function} == 0')

        if not bool(multivector):
            raise MultivectorError(F'Multivector {multivector} is empty')

        if isinstance(multivector, str):
            raise MultivectorError(F'Multivector {multivector} is not a dict')

        len_keys = []
        for e in multivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F'repeated indexes {e} in {multivector}')
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F'invalid key {e} in {multivector}')
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError('keys with different lengths')

        deg_multivector = len(next(iter(multivector)))
        FF = sym.sympify(function)
        multivector = sym.sympify(multivector)

        curl_terms = []
        for e in multivector:
            dict_aux = {tuple(j for j in e if j != i):
                        (-1)**(e.index(i)) * (sym.diff(multivector[e], self.coords[i-1])
                        + (1 / FF) * sym.diff(FF, self.coords[e.index(i)]) * multivector[e]) for i in e}
            curl_terms.append(dict_aux)

        counter = collections.Counter()
        for term in curl_terms:
            counter.update(term)
        curl = dict(counter)
        sym_curl = {e: curl[e] for e in curl}
        str_curl = {e: f"{curl[e]}" for e in curl}

        if deg_multivector - 1 == 0:
            if latex:
                return sym.latex(list(sym_curl.values())[0])
            return list(str_curl.values())[0]

        return sym.latex(sym_curl) if latex else str_curl

    def modular_vf(self, bivector, function, latex=False):
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
        :latex:
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
            >>> pg.modular_vf(bivector, function, latex=True)
            >>> '\\left ( - x_{2} + x_{3}\\right ) \\boldsymbol{Dx}_{1} + \\left ( x_{1} - x_{3}\\right ) \\boldsy
                mbol{Dx}_{2} + \\left ( - x_{1} + x_{2}\\right ) \\boldsymbol{Dx}_{3}'
        """
        # Converts strings to symbolic variables
        bivector = sym.sympify(bivector)
        bivector = {e: (-1) * bivector[e] for e in bivector}
        # Calculate the curl operation
        return self.curl_operator(bivector, function, latex=latex)

    def is_unimodular_homogeneous(self, bivector):
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
            >>> pg.is_unimodular_homogeneous(bivector)
            >>> True
        """
        # Verify that bivector is homogeneous
        for key in bivector:
            if sym.homogeneous_order(bivector[key], *self.coords) is None:
                msg = f'{key}: {bivector[key]} is not a polynomial homogeneous with respect to {self.coordinates}'
                raise Nonhomogeneous(msg)
            if sym.homogeneous_order(bivector[key], *self.coords) < 0:
                msg = f'{key}: {bivector[key]} is not a polynomial homogeneous with respect to {self.coordinates}'
                raise Nonhomogeneous(msg)
        # Finish the new implementation
        mod_vf = self.modular_vf(bivector, '1')
        if not isinstance(mod_vf, dict):
            return mod_vf
        if all(value == '0' for value in mod_vf.values()):
            mod_vf = {}
        return False if bool(mod_vf) else True

    def one_forms_bracket(self, bivector, one_form_1, one_form_2, latex=False):
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
        :latex:
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
            >>> pg.one_forms_bracket(bivector, one_form_1, one_form_2, latex=True)
            >>> 'x_{2} x_{3} \\left(a_{1} b_{2} - a_{1} b_{3} - a_{2} b_{1} + a_{2} b_{3} + a_{3} b_{1} -
                 a_{3} b_{2}\\right) \\boldsymbol{dx}_{1} + x_{1} x_{3} \\left(a_{1} b_{2} - a_{1} b_{3} -
                 a_{2} b_{1} + a_{2} b_{3} + a_{3} b_{1} - a_{3} b_{2}\\right) \\boldsymbol{dx}_{2} +
                 x_{1} x_{2} \\left(a_{1} b_{2} - a_{1} b_{3} - a_{2} b_{1} + a_{2} b_{3} + a_{3} b_{1} -
                 a_{3} b_{2}\\right) \\boldsymbol{dx}_{3}'
        """
        # Validate the bivector
        for e in bivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F'repeated indexes {e} in {bivector}')
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F'invalid key {e} in {bivector}')

        # Check
        if self.is_in_kernel(bivector, one_form_1) and self.is_in_kernel(bivector, one_form_2):
            return {}

        if self.is_in_kernel(bivector, one_form_1):
            form_1_vector = sym.zeros(self.dim + 1, 1)

            for e in bivector:
                if len(set(e)) < len(e):
                    raise MultivectorError(F'repeated indexes {e} in {bivector}')
                if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                    raise MultivectorError(F'invalid key {e} in {bivector}')
            # Verifies the one_form_1 and one_form_2 variables
            for e in one_form_1:
                if e[0] <= 0:
                    raise DiferentialFormError(F'repeated indexes {e} in {one_form_1}')
                form_1_vector[int(*e)] = one_form_1[e]
            for e in one_form_2:
                if e[0] <= 0:
                    raise DiferentialFormError(F'repeated indexes {e} in {one_form_2}')

            form_1_vector = form_1_vector[1:, :]
            jac_form_1 = form_1_vector.jacobian(self.coords)
            d_form_1_mtx = jac_form_1.T - jac_form_1

            sharp_form_2 = self.sharp_morphism(bivector, one_form_2)
            if not isinstance(sharp_form_2, dict):
                return sharp_form_2

            sharp_form_2_vector = sym.zeros(self.dim + 1, 1)
            for e in sharp_form_2:
                sharp_form_2_vector[int(*e)] = sharp_form_2[e]
            sharp_form_2_vector = sharp_form_2_vector[1:, :]

            bracket_vector = d_form_1_mtx * sharp_form_2_vector
            bracket = {(e + 1,): bracket_vector[e] for e in range(self.dim) if sym.simplify(bracket_vector[e]) != 0}

            return sym.latex(bracket) if latex else bracket

        if self.is_in_kernel(bivector, one_form_2):
            form_2_vector = sym.zeros(self.dim + 1, 1)

            for e in bivector:
                if len(set(e)) < len(e):
                    raise MultivectorError(F'repeated indexes {e} in {bivector}')
                if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                    raise MultivectorError(F'invalid key {e} in {bivector}')
            for e in one_form_1:
                if e[0] <= 0:
                    raise DiferentialFormError(F'repeated indexes {e} in {one_form_1}')
            for e in one_form_2:
                if e[0] <= 0:
                    raise DiferentialFormError(F'repeated indexes {e} in {one_form_2}')
                form_2_vector[int(*e)] = one_form_2[e]

            form_2_vector = form_2_vector[1:, :]
            jac_form_2 = form_2_vector.jacobian(self.coords)
            d_form_2_mtx = jac_form_2.T - jac_form_2

            sharp_form_1 = self.sharp_morphism(bivector, one_form_1)
            if not isinstance(sharp_form_1, dict):
                return sharp_form_1

            sharp_form_1_vector = sym.zeros(self.dim + 1, 1)
            for e in sharp_form_1:
                sharp_form_1_vector[int(*e)] = sharp_form_1[e]
            sharp_form_1_vector = sharp_form_1_vector[1:, :]

            bracket_vector = - d_form_2_mtx * sharp_form_1_vector
            bracket = {(e + 1,): bracket_vector[e] for e in range(self.dim) if sym.simplify(bracket_vector[e]) != 0}

            return sym.latex(bracket) if latex else bracket

        form_1_vector = sym.zeros(self.dim + 1, 1)
        form_2_vector = sym.zeros(self.dim + 1, 1)

        for e in one_form_1:
            if e[0] <= 0:
                raise DiferentialFormError(F'repeated indexes {e} in {one_form_1}')
            form_1_vector[int(*e)] = one_form_1[e]
        for e in one_form_2:
            if e[0] <= 0:
                raise DiferentialFormError(F'repeated indexes {e} in {one_form_1}')
            form_2_vector[int(*e)] = one_form_2[e]

        form_1_vector = form_1_vector[1:, :]
        form_2_vector = form_2_vector[1:, :]
        jac_form_1 = form_1_vector.jacobian(self.coords)
        jac_form_2 = form_2_vector.jacobian(self.coords)
        d_form_1_mtx = jac_form_1.T - jac_form_1
        d_form_2_mtx = jac_form_2.T - jac_form_2

        sharp_form_1 = self.sharp_morphism(bivector, one_form_1)
        sharp_form_2 = self.sharp_morphism(bivector, one_form_2)
        if not isinstance(sharp_form_1, dict):
            return sharp_form_1
        if not isinstance(sharp_form_2, dict):
            return sharp_form_2

        sharp_form_1_vector = sym.zeros(self.dim + 1, 1)
        sharp_form_2_vector = sym.zeros(self.dim + 1, 1)
        for e in sharp_form_1:
            sharp_form_1_vector[int(*e)] = sharp_form_1[e]
        for e in sharp_form_2:
            sharp_form_2_vector[int(*e)] = sharp_form_2[e]
        sharp_form_1_vector = sharp_form_1_vector[1:, :]
        sharp_form_2_vector = sharp_form_2_vector[1:, :]

        d_form_2_sharp_1 = d_form_2_mtx * sharp_form_1_vector
        d_form_1_sharp_2 = d_form_1_mtx * sharp_form_2_vector
        pairing_form_2_sharp_1 = (form_2_vector.T * sharp_form_1_vector)[0]
        d_pairing_form_2_sharp_1 = sym.Matrix(sym.derive_by_array(pairing_form_2_sharp_1, self.coords))

        bracket_vector = - d_form_2_sharp_1 + d_form_1_sharp_2 + d_pairing_form_2_sharp_1
        sym_bracket = {(e + 1,): bracket_vector[e] for e in range(self.dim) if sym.simplify(bracket_vector[e]) != 0}
        str_bracket = {(e + 1,): f"{bracket_vector[e]}" for e in range(self.dim) if sym.simplify(bracket_vector[e]) != 0}  # noqa:E501
        return sym.latex(sym_bracket) if latex else str_bracket

    def gauge_transformation(self, bivector, two_form, gauge_biv=True, det=False, latex=False):
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
            >>> P = {(1,2): "P12", (1,3): "P13", (2,3): "P23"}
            >>> # For two form L12*dx1^dx2 + L13*dx1^dx3 + L23*dx2^dx3
            >>> lambda_ = {(1,2): "L12", (1,3): "L13", (2,3): "L23"}
            >>> (gauge_bivector, determinant) = pg.gauge_transformation(P, lambda)
            >>> gauge_bivector
            >>> {  (1, 2): P12/(L12*P12 + L13*P13 + L23*P23 + 1),
                   (1, 3): P13/(L12*P12 + L13*P13 + L23*P23 + 1),
                   (2, 3): P23/(L12*P12 + L13*P13 + L23*P23 + 1)
                }
            >>> determinant
            >>> (L12*P12 + L13*P13 + L23*P23 + 1)**2
        """
        if not gauge_biv and not det:
            return {}

        bivector_matrix = self.bivector_to_matrix(bivector)
        two_form_matrix = self.bivector_to_matrix(two_form)
        if not isinstance(bivector_matrix, sym.matrices.dense.MutableDenseMatrix):
            return bivector_matrix
        if not isinstance(two_form_matrix, sym.matrices.dense.MutableDenseMatrix):
            return two_form_matrix

        I_plus_deltas = [sym.eye(self.dim) + two_form_matrix.col(0) * ((-1) * bivector_matrix.row(0))]
        for k in range(1, self.dim - 1):
            I_plus_deltas.append(I_plus_deltas[k - 1] + two_form_matrix.col(k) * ((-1) * bivector_matrix.row(k)))
        adj_I_deltas = [sym.Matrix.adjugate(e) for e in I_plus_deltas]
        viT_adj_I_deltas_ui = [(((-1) * bivector_matrix.row(i)) * adj_I_deltas[i-1] * two_form_matrix.col(i))[0] for i in range(1, self.dim)]  # noqa: E501
        sum_viT_adj_I_deltas_ui = sum(viT_adj_I_deltas_ui)
        gauge_det = 1 + (((-1) * bivector_matrix.row(0)) * two_form_matrix.col(0))[0] + sum_viT_adj_I_deltas_ui

        if det and not gauge_biv:
            if latex:
                return sym.latex(gauge_det)
            return f"{gauge_det}"

        gauge_det = sym.simplify(gauge_det)
        if gauge_det == 0:
            return False

        BP = two_form_matrix * bivector_matrix
        I_minus_BP = sym.eye(self.dim) - BP
        adj_I_BP = sym.Matrix.adjugate(I_minus_BP)
        inv_I_BP = (1 / gauge_det) * adj_I_BP
        gauge_matrix = bivector_matrix * inv_I_BP

        gauge_matrix = sym.matrices.SparseMatrix(gauge_matrix)
        gauge_matrix_RL = gauge_matrix.RL
        sym_gauge_bivector = {(e[0] + 1, e[1] + 1): e[2] for e in gauge_matrix_RL if e[0] < e[1]}
        str_gauge_bivector = {(e[0] + 1, e[1] + 1): f"{e[2]}" for e in gauge_matrix_RL if e[0] < e[1]}

        if det:
            if latex:
                return sym.latex(sym_gauge_bivector), sym.latex(gauge_det)
            return str_gauge_bivector, f"{gauge_det}"
        if latex:
            return sym.latex(sym_gauge_bivector)
        return str_gauge_bivector

    def flaschka_ratiu_bivector(self, casimirs, symplectic_form=False, latex=False):
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
        :latex:
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
            >>> pg.flaschka_ratiu_bivector(casimirs, symplectic_form=False, latex=False)
            >>> {(1, 2): -2*x3, (1, 3): 2*x2, (2, 3): -2*x1}
            >>> pg.flaschka_ratiu_bivector(casimirs, symplectic_form=False, latex=True)
            >>> '-2*x3*Dx1^Dx2 + 2*x2*Dx1^Dx3 - 2*x1*Dx2^Dx3'
            >>> bivector, symplectic_form = pg.flaschka_ratiu_bivector(casimirs, symplectic_form=True,
                                                                       latex=False)
            >>> bivector
            >>> {(1, 2): -2*x3, (1, 3): 2*x2, (1, 4): 0, (2, 3): -2*x1, (2, 4): 0, (3, 4): 0}
            >>> symplectic_form
            >>> x3*Dx1^Dx2/(2*(x1**2 + x2**2 + x3**2)) - x2*Dx1^Dx3/(2*x1**2 + 2*x2**2 + 2*x3**2) + x1*Dx2^Dx3/(2*(x1**2 + x2**2 + x3**2)  # noqa
            >>> bivector, symplectic_form = pg.flaschka_ratiu_bivector(casimirs, symplectic_form=True, latex=True)  # noqa
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
        # Validate the dimension and the Casimir's functions
        if sym.simplify(self.dim - 2) <= 0:
            raise DimensionError(F'The dimension {self.dim} is minor to 3')
        if len(casimirs) != self.dim - 2:
            raise CasimirError(F"The length to Casimir's functions is distinct to {self.dim - 2}")
        # Convert the Casimir's functions to symbolic type
        casimirs = sym.sympify(casimirs)
        # Calculate the Poisson bivector
        grad_casims_matrix = sym.Matrix([sym.derive_by_array(e, self.coords) for e in casimirs])
        indixes = itls.combinations(range(self.dim), 2)
        FR_matrices = {(e[0] + 1, e[1] + 1): del_columns(grad_casims_matrix, e) for e in indixes}
        FR_bivector = {e: (-1)**(sum(e)) * FR_matrices[e].det(method='LU') for e in FR_matrices}
        sym_FR_bivector = {e: FR_bivector[e] for e in FR_bivector if sym.simplify(FR_bivector[e]) != 0}
        str_FR_bivector = {e: f"{FR_bivector[e]}" for e in FR_bivector if sym.simplify(FR_bivector[e]) != 0}

        # Calculate the symplectic form
        if symplectic_form:
            norm = sum([e**2 for e in FR_bivector.values()])
            sym_symp_form = {e: (-1) * (1/norm) * FR_bivector[e] for e in FR_bivector}
            str_symp_form = {e: f"{(-1) * (1/norm) * FR_bivector[e]}" for e in FR_bivector}
            if latex:
                return sym.latex(sym_FR_bivector), sym.latex(sym_symp_form)
            return str_FR_bivector, str_symp_form

        if latex:
            return sym.latex(sym_FR_bivector)
        return str_FR_bivector

    def linear_normal_form_R3(self, bivector, latex=False):
        """ Calculates a normal form for Lie-Poisson bivector fields on R^3 modulo linear isomorphisms.

        Parameters
        ==========
        :bivector:
            Is a Lie-Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :latex:
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
            >>> pg.linear_normal_form_R3(bivector, latex=True)
            >>> '\\left ( 4 a x_{2} + x_{1}\\right ) \\boldsymbol{Dx}_{1}\\wedge \\boldsymbol{Dx}_{3} +
                 \\left ( 4 a x_{1} + x_{2}\\right ) \\boldsymbol{Dx}_{2}\\wedge \\boldsymbol{Dx}_{3}'
        """
        # Verify that bivector is homogeneous
        for key in bivector:
            if sym.homogeneous_order(bivector[key], *self.coords) is None:
                msg = f'{key}: {bivector[key]} is not a linear polynomial with respect to {self.coordinates}'
                raise Nonlinear(msg)

            if sym.homogeneous_order(bivector[key], *self.coords) != 1:
                msg = f'{key}: {bivector[key]} is not a linear polynomial with respect to {self.coordinates}'
                raise Nonlinear(msg)

        # Verifies the bivector
        if not bool(bivector):
            return {}
        for e in bivector:
            if len(e) != 2 or len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F'invalid key {e} in {bivector}')
            if len(set(e)) < len(e):
                raise MultivectorError(F'repeated indexes {e} in {bivector}')

        if [bivector[e].find(f'{self.variable}') for e in bivector].count(-1) != 0:
            return {}

        bivector = sym.sympify(bivector)
        pair_E_P = []
        for e in bivector:
            if sym.simplify(e[0] * e[1] - 6) == 0:
                pair_E_P.append(-self.coords[0] * bivector[e])
            if sym.simplify(e[0] * e[1] - 3) == 0:
                pair_E_P.append(self.coords[1] * bivector[e])
            if sym.simplify(e[0] * e[1] - 2) == 0:
                pair_E_P.append(-self.coords[2] * bivector[e])
        pair_E_P = sym.sympify('1/2') * sum(pair_E_P)
        hessian_pair_E_P = sym.sympify('1/2') * sym.hessian(pair_E_P, self.coords[:3])
        rank_hess = hessian_pair_E_P.rank()
        egvals_hess = [sym.re(e) for e in hessian_pair_E_P.eigenvals(rational=True, multiple=True)]
        sign_hess = sum([sym.sign(e) for e in egvals_hess])

        if self.is_unimodular_homogeneous(bivector):
            if sym.simplify(rank_hess - 3) == 0:
                if sym.simplify(sym.Abs(sign_hess) - 3) == 0:
                    if latex:
                        return sym.latex({
                            (1, 2): F'{self.coords[2]}',
                            (1, 3): F'-{self.coords[1]}',
                            (2, 3): F'{self.coords[0]}'
                        })
                    return {
                        (1, 2): F'{self.coords[2]}',
                        (1, 3): F'-{self.coords[1]}',
                        (2, 3): F'{self.coords[0]}',
                    }  # 'rank 3, sum 3,-3'

                if sym.simplify(sym.Abs(sign_hess) - 1) == 0:
                    if latex:
                        return sym.latex({
                            (1, 2): F'-{self.coords[2]}',
                            (1, 3): F'-{self.coords[1]}',
                            (2, 3): F'{self.coords[0]}',
                        })
                    return {
                        (1, 2): F'-{self.coords[2]}',
                        (1, 3): F'-{self.coords[1]}',
                        (2, 3): F'{self.coords[0]}',
                    }  # 'rank 3, sum 1,-1'

            if sym.simplify(rank_hess - 2) == 0:
                if sym.simplify(sym.Abs(sign_hess) - 2) == 0:
                    if latex:
                        return sym.latex({(1, 3): F'-{self.coords[1]}', (2, 3): F'{self.coords[0]}'})
                    return {(1, 3): F'-{self.coords[1]}', (2, 3): F'{self.coords[0]}'}  # 'rank 2, sum 2,-2'

                if sign_hess == 0:
                    if latex:
                        return sym.latex({(1, 3): F'{self.coords[1]}', (2, 3): F'{self.coords[0]}'})
                    return {(1, 3): F'{self.coords[1]}', (2, 3): F'{self.coords[0]}'}  # 'rank 2, sum 0'

            if sym.simplify(rank_hess - 1) == 0:
                if latex:
                    return sym.latex({(2, 3): F'{self.coords[0]}'})
                return {(2, 3): F'{self.coords[0]}'}  # 'rank 1'

        if rank_hess == 0:
            if latex:
                return sym.latex({(1, 3): F'{self.coords[0]}', (2, 3): F'{self.coords[1]}'})
            return {(1, 3): F'{self.coords[0]}', (2, 3): F'{self.coords[1]}'}  # 'rank 0'

        if sym.simplify(rank_hess - 2) == 0:
            if sym.simplify(sym.Abs(sign_hess) - 2) == 0:
                if latex:
                    return sym.latex({
                        (1, 3): F'{self.coords[0]} - 4*a*{self.coords[1]}',
                        (2, 3): F'4*a*{self.coords[0]} + {self.coords[1]}',
                    })
                return {
                    (1, 3): F'{self.coords[0]} - 4*a*{self.coords[1]}',
                    (2, 3): F'4*a*{self.coords[0]} + {self.coords[1]}',
                }  # 'rank 2, sum 2,-2'

            if sign_hess == 0:
                if latex:
                    return sym.latex({
                        (1, 3): F'{self.coords[0]} + 4*a*{self.coords[1]}',
                        (2, 3): F'4*a*{self.coords[0]} + {self.coords[1]}'
                    })
                return {
                    (1, 3): F'{self.coords[0]} + 4*a*{self.coords[1]}',
                    (2, 3): F'4*a*{self.coords[0]} + {self.coords[1]}'
                }  # 'rank 2, sum 0'

        if sym.simplify(rank_hess - 1) == 0:
            if latex:
                return sym.latex({(1, 3): f'{self.coords[0]}', (2, 3): f'4*{self.coords[0]} + {self.coords[1]}'})
            return {(1, 3): f'{self.coords[0]}', (2, 3): f'4*{self.coords[0]} + {self.coords[1]}'}  # 'rank 1'

        return {}

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
        # Calculate the linear normal forms of the bivectors
        normal_1 = self.linear_normal_form_R3(bivector_1)
        normal_2 = self.linear_normal_form_R3(bivector_2)

        if not bool(normal_1) and not bool(normal_2):
            return True

        if not isinstance(normal_1, dict):
            return normal_1
        if not isinstance(normal_2, dict):
            return normal_2

        if len(normal_1) - len(normal_2) != 0:
            return False

        if len(normal_1) - 1 == 0 and len(normal_2) - 1 == 0:
            return True

        # Simplifies the linear normal forms of the bivectors
        normal_1 = sym.sympify(normal_1)
        normal_2 = sym.sympify(normal_2)

        if len(normal_1) - 2 == 0 and len(normal_2) - 2 == 0:
            compare = [1 for e in normal_1 if sym.simplify(normal_1[e] - normal_2[e]) != 0]
            return True if len(compare) == 0 else False

        if len(normal_1) - 3 == 0 and len(normal_2) - 3 == 0:
            return True if sym.simplify(normal_1[(1, 2)] - normal_2[(1, 2)]) == 0 else False

        return False

    def coboundary_operator(self, bivector, multivector, latex=False):
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
        :latex:
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
            >>> pg.coboundary_operator(bivector, multivector, latex=False)
            >>> {}
            >>> pg.coboundary_operator(bivector, multivector latex=True)
            >>> '{}'
        """
        if not bool(multivector):
            return {}

        if isinstance(multivector, str):
            # [P,f] = -X_f, for any function f.
            return self.hamiltonian_vf(bivector, f'(-1) * {multivector}', latex=latex)

        # Verifies the bivector param
        len_keys = []
        for e in bivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F'repeated indexes {e} in {bivector}')
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F'invalid key {e} in {bivector}')
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError(F'keys with different lengths in {bivector}')

        # Verifies the multivector param
        len_keys = []
        for e in multivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F'repeated indexes {e} in {bivector}')
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F'invalid key {e} in {bivector}')
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError(F'keys with different lengths in {bivector}')

        # Degree of multivector
        deg_mltv = len(next(iter(multivector)))
        if deg_mltv + 1 > self.dim:
            return {}

        mltv = sym.sympify(multivector)
        # A dictionary for the first term of [P,A], A a multivector
        schouten_biv_mltv_1 = {}
        lich_poiss_aux_1 = 0
        # The first term of [P,A] is a sum of Poisson brackets {xi,A^J}
        range_list = range(1, self.dim + 1)
        for z in itls.combinations(range_list, deg_mltv + 1):
            for i in z:
                # List of indices for A without the index i
                nw_idx = [e for e in z if e != i]
                # Calculates the Poisson bracket {xi,A^J}, J == nw_idx
                mltv_nw_idx = mltv.get(tuple(nw_idx), 0)
                lich_poiss_aux_11 = (-1)**(z.index(i)) * sym.sympify(self.poisson_bracket(bivector, self.coords[i - 1], mltv_nw_idx))  # noqa: E501
                lich_poiss_aux_1 = lich_poiss_aux_1 + lich_poiss_aux_11
            # Add the brackets {xi,A^J} in the list schouten_biv_mltv_1
            schouten_biv_mltv_1.update({z: lich_poiss_aux_1})
            lich_poiss_aux_1 = 0

        bivector = sym.sympify(bivector)
        # A dictionary for the second term of [P,A]
        schouten_biv_mltv_2 = {}
        sum_i = 0
        sum_y = 0
        # Second term of [P,A] is a sum of Dxk(P^iris)*A^((J+k)\iris))
        for z in itls.combinations(range_list, deg_mltv + 1):
            for y in itls.combinations(z, 2):
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
                        sum_i_aux = (-1)**(z.index(y[0]) + z.index(y[1]) + nw_idx_mltv.index(i)) * sym.diff(bivector_y, self.coords[i - 1]) * mltv_nw_idx_mltv  # noqa: E501
                        sum_i = sum_i + sum_i_aux
                sum_y_aux = sum_i
                sum_y = sum_y + sum_y_aux
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
            schouten_biv_mltv.update({ky: schouten_biv_mltv_1[ky] + schouten_biv_mltv_2[ky]})
        sym_schouten_biv_mltv = {e: schouten_biv_mltv[e] for e in schouten_biv_mltv if sym.simplify(schouten_biv_mltv[e]) != 0}  # noqa: E501
        str_schouten_biv_mltv = {e: f"{schouten_biv_mltv[e]}" for e in schouten_biv_mltv if sym.simplify(schouten_biv_mltv[e]) != 0}  # noqa: E501

        return sym.latex(sym_schouten_biv_mltv) if latex else str_schouten_biv_mltv

    def jacobiator(self, bivector, latex=False):
        """ Calculates de Schouten-Nijenhuis bracket of a given bivector field with himself, that is [P,P]
            where [,] denote the Schouten bracket for multivector fields.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :latex:
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
            >>> pg.jacobiator(bivector, latex=False)
            >>> {}
            >>> pg.jacobiator(bivector, latex=True)
            >>> '\\left\\{ \\right\\}'
        """
        jac = self.coboundary_operator(bivector, bivector, latex=latex)
        if not isinstance(jac, dict):
            return jac
        return jac

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
            >>> pg.is_poisson_bivector(bivector)
            >>> True
        """
        jac = self.coboundary_operator(bivector, bivector)
        if not isinstance(jac, dict):
            return jac
        if all(value == '0' for value in jac.values()):
            jac = {}
        return False if bool(jac) else True

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
        sch_biv_vf = self.coboundary_operator(bivector, vector_field)
        if not isinstance(sch_biv_vf, dict):
            return sch_biv_vf
        if all(value == '0' for value in sch_biv_vf.values()):
            sch_biv_vf = {}
        return False if bool(sch_biv_vf) else True

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
        sch_biv1_biv2 = self.coboundary_operator(bivector_1, bivector_2)
        if not isinstance(sch_biv1_biv2, dict):
            return sch_biv1_biv2
        if all(value == '0' for value in sch_biv1_biv2.values()):
            sch_biv1_biv2 = {}
        return False if bool(sch_biv1_biv2) else True
