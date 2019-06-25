import sympy as sym
import itertools as itools


class PoissonGeometry:
    """ Class for Poisson Geometry.

        This class provides some useful tools for Poisson-Nijenhuis
        calculus on Poisson manifolds.
    
    """

    def __init__(self, dimension, variable='x'):
        # Obtains the dimension
        self.dim = dimension
        # Create the symbolics symbols
        self.coordinates = sym.symbols(f'{variable}1:{self.dim + 1}')


    def bivector_to_matrix(self, bivector):
        """ Constructs the matrix of a 2-contravariant tensor field or
        bivector field.

        Remark:
        The associated matrix of a bivector field
            P = Pij*Dxi^Dxj, 1 <= i < j <= dim,
        is defined by [Pij].

        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.

        :return: a symbolic skew-symmetric matrix of dimension
         (dim)x(dim). For the previous example,
            Matrix([[0, x3, -x2], [-x3, 0, x1], [x2, -x1, 0]]).
        """
        # Creates a symbolic Matrix
        biv_mtx = sym.MatrixSymbol('P', self.dim, self.dim)
        biv_mtx = sym.Matrix(biv_mtx)

        # Assigns the corresponding coefficients of the bivector field
        for z in itools.combinations_with_replacement(range(1, self.dim + 1), 2):
            i , j = z[0], z[1]
            biv_mtx[i-1,j-1] = 0 if i == j else sym.sympify(bivector[int(''.join(str(i) for i in z))])
            biv_mtx[j-1,i-1] = (-1) * biv_mtx[i-1, j-1]

        # Return a symbolic skew-symmetric matrix
        return biv_mtx


    def sharp_map(self, bivector, one_form):
        """ Calculates the image of a differential 1-form under the
        vector bundle morphism 'sharp' induced by a (Poisson) bivector
        field.

        Remark:
        The morphism 'sharp', P#: T*M -> TM,
        is defined by
            P#(alpha) := i_(alpha)P,
        where P is a (Poisson) bivector field on a manifold M, alpha a 1-form on M and
        i the interior product of alpha and P.

        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.

        :param one_form: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {1: 'a1*x1', 2: 'a2*x2', 3: 'a3*x3'}.
         The 'keys' are the indices of the coefficients of a given
         differential 1-form alpha and the 'values' their coefficients.
         In this case,
            alpha = a1*x1*dx1 + a2*x2*dx2 + a3*x3*dx3.

        :return: a dictionary with integer type 'keys' and symbol type
         'values'. For the previous example,
            {1: x2*(-a2*x3 + a3*x3x), 2: x1*(a1*x3 - a3*x3x), 3: x1*x2*(-a1 + a2)},
         which represents the vector field
            P#(alpha) = x2*(-a2*x3 + a3*x3x)*Dx1 + x1*(a1*x3 - a3*x3x)*Dx2 + x1*x2*(-a1 + a2)*Dx3.
        """
        # Converts strings to symbolic variables
        for key in one_form.keys():
            one_form[key] = sym.sympify(one_form[key])
        for key in bivector.keys():
            bivector[key] = sym.sympify(bivector[key])

        """
            Calculation of the vector field bivector^#(one_form) as
                ∑_{i}(∑_{j} bivector_ij (one_form_i*Dx_j - one_form_j*Dx_i)
            where i < j.
        """
        p_sharp_aux = [0] * self.dim
        for ij in bivector.keys():
            # Converts the key in a tuple
            tuple_ij = tuple(map(int, str(ij)))
            # Calculates one_form_i*Pij*Dxj
            p_sharp_aux[tuple_ij[1] - 1] = sym.simplify(
                p_sharp_aux[tuple_ij[1] - 1] + one_form[tuple_ij[0]] * bivector[ij])
            # Calculates one_form_j*Pji*Dxi
            p_sharp_aux[tuple_ij[0] - 1] = sym.simplify(
                p_sharp_aux[tuple_ij[0] - 1] - one_form[tuple_ij[1]] * bivector[ij])
        # Creates a dictionary which represents the vector field P#(alpha)
        p_sharp_dict = dict(zip(range(1, self.dim + 1), p_sharp_aux))

        return p_sharp_dict


    def is_in_kernel(self, bivector, one_form):
        """ Check if a differential 1-form belongs to the kernel of a
        given (Poisson) bivector field.

        Remark:
        A diferencial 1-form alpha on a Poisson manifold (M,P) belong to the kernel of P if
            P#(alpha) = 0,
        where P#: T*M -> TM is the vector bundle morphism defined by
            P#(alpha) := i_(alpha)P,
        with i the interior product of alpha and P.

        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x3', 13: '-x2', 23: 'x1'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P = x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3.

        :param one_form: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {1: 'x1', 2: 'x2', 3: 'x3'}.
         The 'keys' are the indices of the coefficients of a given
         differential 1-form alpha and the 'values' their coefficients.
         In this case,
            alpha = x1*dx1 + x2*dx2 + x3*dx3.

        :return: a boolean variable.
        """
        # Check if the vector field bivector#(one_form) is zero or not
        if all(val == 0 for val in self.sharp_map(bivector, one_form).values()):
            return True
        return False


    def hamiltonian_vector_field(self, bivector, hamiltonian_function):
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
        ham_vector_field = self.sharp_map(bivector, dict(zip(range(1, self.dim + 1), dh)))

        # return a dictionary
        return ham_vector_field


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


    def poisson_bracket(self, bivector, function_1, function_2):
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
        if f1 == f2:
            return 0
        # Calculates the differentials of function_1 and function_2
        df1 = dict(zip(range(1, self.dim + 1), sym.derive_by_array(f1, self.coordinates)))
        df2 = dict(zip(range(1, self.dim + 1), sym.derive_by_array(f2, self.coordinates)))
        # Calculates {f1,f2} = <df2,P#(df1)> = ∑_{i} (P#(df1))^i * (df2)_i
        bracket_f1_f2 = sym.simplify(sum(self.sharp_map(bivector, df1)[index] * df2[index] for index in df1))

        # Return a symbolic type expression
        return bracket_f1_f2


    def lichnerowicz_poisson_operator(self, bivector, multivector):
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
            if next(iter(multivector)) == 0:  # In this case, mltv is a fnction
                # [P,f] = -X_f, for any function f.
                schouten_biv_f = self.hamiltonian_vector_field(
                    bivector, (-1) * sym.sympify(
                        multivector[next(iter(multivector))]))
                return schouten_biv_f
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
                return schouten_biv_mltv


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


    def modular_vector_field(self, bivector, function):
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
        for key in bivector.keys():
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
        return modular_vf_a  # Return a dictionary


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


    def one_forms_bracket(self, bivector, one_form_1, one_form_2):
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
                ii_sharp_alpha_d_beta[z[0] - 1] + self.sharp_map(
                    bivector, one_form_1)[z[1]] * (sym.diff(
                    one_form_2[z[0]], self.coordinates[z[1] - 1]) - sym.diff(
                    one_form_2[z[1]], self.coordinates[z[0] - 1])))
            ii_sharp_alpha_d_beta[z[1] - 1] = sym.simplify(
                ii_sharp_alpha_d_beta[z[1] - 1] + self.sharp_map(
                    bivector, one_form_1)[z[0]] * (sym.diff(
                    one_form_2[z[1]], self.coordinates[z[0] - 1]) - sym.diff(
                    one_form_2[z[0]], self.coordinates[z[1] - 1])))
            ii_sharp_beta_d_alpha[z[0] - 1] = sym.simplify(
                ii_sharp_beta_d_alpha[z[0] - 1] + self.sharp_map(
                    bivector, one_form_2)[z[1]] * (sym.diff(
                    one_form_1[z[0]], self.coordinates[z[1] - 1]) - sym.diff(
                    one_form_1[z[1]], self.coordinates[z[0] - 1])))
            ii_sharp_beta_d_alpha[z[1] - 1] = sym.simplify(
                ii_sharp_beta_d_alpha[z[1] - 1] + self.sharp_map(
                    bivector, one_form_2)[z[0]] * (sym.diff(
                    one_form_1[z[1]], self.coordinates[z[0] - 1]) - sym.diff(
                    one_form_1[z[0]], self.coordinates[z[1] - 1])))
        # Calculates the third term of the Lie bracket
        # d_P(alpha,beta) = d(<beta,P#(alpha)>), with <,> the pairing
        # Formula: d(<beta,P#(alpha)>) = d(P#(alpha)^i * beta_i)
        d_pairing_beta_sharp_alpha = sym.simplify(sym.derive_by_array(sum(
            one_form_2[ky] * self.sharp_map(bivector, one_form_1)[ky] for ky
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
        return one_forms_brack  # Return a dictionary


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
        x_aux = sym.symbols(f'x1:{4}')
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


    def quadratic_normal_form_R3(self, bivector):
        """ Calculates a normal form for quadratic Poisson bivector
        fields on R^3 modulo linear isomorphisms.
        Recall that two quadratic bivector fields P1 and P2 on R^3 is
        said to be equivalents if there exists a linear isomorphism
        T:R^3 -> R^3 such that T*P2 = P1.
        :param bivector: is a dictionary with integer type 'keys' and
         string type 'values'. For example, on R^3,
            {12: 'x1*x2', 13: '-x1*x3', 23: 'x2*x3'}.
         The 'keys' are the ordered indices of the given bivector field
         P and the 'values' their coefficients. In this case,
            P1 = x1*x2*Dx1^Dx2 - x1*x3*Dx1^Dx3 + x2*x3*Dx2^Dx3.
        :return: a dictionary with integer type 'keys' and
         string type 'values'. For the previous example,
            {12: 'x1*x2', 13: '-x1*x3', 23: 'x2*x3'},
         which represents the normal form
            Q = x1*x2*Dx1^Dx2 - x1*x3*Dx1^Dx3 + x2*x3*Dx2^Dx3.
        """
        x_aux = sym.symbols(f'x1:{4}')
        mtx_mod_vf_biv = sym.Matrix((sym.derive_by_array(
            self.modular_vector_field(bivector, 1)[1], x_aux), sym.derive_by_array(
            self.modular_vector_field(bivector, 1)[2], x_aux), sym.derive_by_array(
            self.modular_vector_field(bivector, 1)[3], x_aux)))
        egval_mod_vf = mtx_mod_vf_biv.eigenvals()
        # All multiplicity are 1
        if mtx_mod_vf_biv == sym.zeros(3):
            # print(f'Case I')
            return bivector
        if reduce((lambda a, b: a * b), list(egval_mod_vf.values())) == 1:
            # Three real eigenvalues
            if all(sy.im(t) == 0 for t in list(egval_mod_vf.keys())):
                # Nonzero eigenvalues
                if reduce((lambda c, d: c * d),
                          list(egval_mod_vf.keys())) != 0:
                    # print(f'Case II', egval_mod_vf)
                    return {12: sym.sympify(f'(1/3*(b1 - b2) - a)*x1*x2'),
                            13: sym.sympify(f'(1/3*(b1 - b3) + a)*x1*x3'),
                            23: sym.sympify(f'(1/3*(b2 - b3) - a)*x2*x3')}
                # One zero eigenvalue
                else:
                    print(f'Case III', egval_mod_vf)
            # Two complex and one real eigenvalues
            else:
                # Nonzero eigenvalues
                if reduce((lambda c, d: c * d), list(egval_mod_vf.keys())) != 0:
                    # print(f'Case V', egval_mod_vf)
                    return {12: sym.sympify(f'-(a + 1/3*b2)*(x1**2 + x2**2)'),
                            13: sym.sympify(f'(b1*x1 + (2*a - 1/3*b2)*x2)*x3'),
                            23: sym.sympify(f'(b1*x2 - (2*a - 1/3*b2)*x1)*x3')}
                # One zero eigenvalue
                else:
                    print(f'Case VI', egval_mod_vf)
        # One nonzero real eigenvalue with multiplicity 2
        if reduce((lambda a, b: a * b), list(egval_mod_vf.values())) == 2:
            # Diagonalizable case
            if mtx_mod_vf_biv.is_diagonalizable(reals_only=True):
                print(f'Case IV')
            # Non-diagonalizable case
            else:
                # print(f'Case VII')
                return {12: sym.sympify(f'(1/3 - a)*x2**2'),
                        13: sym.sympify(f'(b*x1 + (2*a + 1/3)*x2)*x3'),
                        23: sym.sympify(f'b*x2*x3')}
        # Zero eigenvalue with multiplicity 3
        if reduce((lambda a, b: a * b), list(egval_mod_vf.values())) == 3:
            # Rank 2 case
            if mtx_mod_vf_biv.rank() == 2:
                print(f'Case IX')
            # Rank 1 case
            if mtx_mod_vf_biv.rank() == 1:
                print(f'Case VIII')


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
        I_minus_BP = sym.Matrix(sy.simplify(
            sy.eye(self.dim) - two_form_B * self.bivector_to_matrix(bivector)))
        det_I_minus_BP = sym.factor(sym.simplify(I_minus_BP.det()))
        if det_I_minus_BP == 0:
            return f'\nThe differential 2-form can\'t induces a gauge ' \
                f'transformation.'
        else:
            gauge_mtx = sym.Matrix(
                sym.simplify(self.bivector_to_matrix(bivector) * (I_minus_BP.inv())))
            return gauge_mtx, sym.sympify(det_I_minus_BP)

# CLASSIFICATION

    def lie_der_vf_2_form(self, vector_field, two_form):
        for i in vector_field:
            vector_field[i] = sym.sympify(vector_field[i])
        for key in two_form:
            two_form[key] = sym.sympify(two_form[key])
        lie_der_2_form = {}
        for key in two_form:
            lie_der_aux_k = sym.simplify(sum(
                vector_field[k] * sym.diff(two_form[key], self.coordinates[k - 1])
                for k in range(1, self.dim + 1)))
            ky2 = tuple(int(i) for i in str(key))
            lie_der_aux_ki = sym.simplify(sum(
                (-1) * sym.diff(vector_field[k], self.coordinates[ky2[1] - 1]) *
                two_form[int(''.join((f'{k}', f'{ky2[0]}')))] for k in
                range(1, ky2[0])))
            lie_der_aux_ik = sym.simplify(sum(
                sym.diff(vector_field[k], self.coordinates[ky2[1] - 1]) * two_form[
                    int(''.join((f'{ky2[0]}', f'{k}')))] for k in
                range(ky2[0] + 1, self.dim + 1)))
            lie_der_aux_kj = sym.simplify(sum(
                sym.diff(vector_field[k], self.coordinates[ky2[0] - 1]) * two_form[
                    int(''.join((f'{k}', f'{ky2[1]}')))] for k in
                range(1, ky2[1])))
            lie_der_aux_jk = sym.simplify(sum(
                (-1) * sym.diff(vector_field[k], self.coordinates[ky2[0] - 1]) *
                two_form[int(''.join((f'{ky2[1]}', f'{k}')))] for k in
                range(ky2[1] + 1, self.dim + 1)))
            lie_der_2_form.setdefault(key, sym.simplify(
                lie_der_aux_k + lie_der_aux_ki + lie_der_aux_ik
                + lie_der_aux_kj + lie_der_aux_jk))
        return lie_der_2_form


    def ext_der_two_form(self, two_form):
        d_2_form = {}
        for z in itools.combinations(range(1, self.dim + 1), 3):
            d_2_form_aux = sym.simplify(sym.simplify(sym.diff(
                two_form[int(''.join((f'{z[1]}', f'{z[2]}')))],
                self.coordinates[z[0] - 1])) + sym.simplify((-1) * sym.diff(
                two_form[int(''.join((f'{z[0]}', f'{z[2]}')))],
                self.coordinates[z[1] - 1])) + sym.simplify(sym.diff(
                two_form[int(''.join((f'{z[0]}', f'{z[1]}')))],
                self.coordinates[z[2] - 1])))
            d_2_form.setdefault(int(''.join(str(i) for i in z)),
                                sym.simplify(d_2_form_aux))
        return d_2_form


    def inner_euler_vf_2_form(self, two_form):
        for key in two_form:
            two_form[key] = sym.sympify(two_form[key])
        ii_E_two_form = {}
        for i in range(1, self.dim + 1):
            ii_E_2_form_ji = sum(self.coordinates[j - 1] * two_form[int(''.join((f'{j}', f'{i}')))] for j in range(1, i))
            ii_E_2_form_ij = sum((-1) * self.coordinates[j - 1] * two_form[int(''.join((f'{i}', f'{j}')))] for j in range(i + 1, self.dim + 1))
            ii_E_two_form.setdefault(i, sym.simplify(ii_E_2_form_ji + ii_E_2_form_ij))
        return ii_E_two_form


    def inner_euler_vf_2_form_2(self, two_form):
        form = sym.MatrixSymbol('O', self.dim, self.dim)
        form = sym.Matrix(form)
        # Assigns the corresponding coefficients of the 2-form
        for z in itools.combinations_with_replacement(range(1, self.dim + 1),
                                                      2):
            if z[0] == z[1]:
                form[z[0] - 1, z[1] - 1] = 0
            else:
                form[z[0] - 1, z[1] - 1] = sym.sympify(
                    two_form[int(''.join(str(i) for i in z))])
                form[z[1] - 1, z[0] - 1] = (-1) * form[
                    z[0] - 1, z[1] - 1]
        euler_vf = sym.Matrix(self.coordinates)
        ii_E_2_form_2 = (-1) * form * euler_vf
        return ii_E_2_form_2
