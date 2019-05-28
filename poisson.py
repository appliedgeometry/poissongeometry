# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sympy
import itertools
from galgebra.ga import Ga as ga
from utils import matrix_to_dictionary, complete_args, Dx_basis



class PoissonGeometry(object):

    def poisson_bracket(self, matrix, functions=[]):
        ''' This method calculates the Poisson bracket associates with Poisson matrix (bivector) π
        Parametres:
            * symbolic matrix π
            * list with two functions [f, g], where f and g are strings, and this method converts it in symbolic variables
        return:
            {f,g}_π, that is a symbol variable
        '''
        # Verify if functions contain two elements
        if len(functions) != 2:
            print("You array not contain two functions")
            return None
        else:
            # Convert matrix to dictionary
            dictionary_with_coeff = matrix_to_dictionary(matrix)
            # Obtains all symbolic variables from the matrix
            args_matrix = dictionary_with_coeff.pop('args')
            variables = complete_args(args_matrix)
            # Obtains all the combinations from n in 2
            combinations = [x for x in itertools.combinations(range(1, len(variables)+1), 2)]
            # Convert the string functions to symbolic functions
            functions = [sympy.sympify(functions[0]), sympy.sympify(functions[1])]
            # Calculates the poisson bracket
            bracket_result = 0
            for combination in combinations:
                # Convert (i,j) to 'ij'
                ij_tuple_str = tuple(str(element) for element in combination)
                ij = ''.join(ij_tuple_str)
                # Calculates {f,g}_π = Σ(π_{ij} (∂f/∂x_i * ∂g/∂x_j - ∂g/∂x_i * ∂f/∂x_j)
                bracket_result = bracket_result + dictionary_with_coeff[ij]*(sympy.diff(functions[0], variables[combination[0]-1]) * sympy.diff(functions[1], variables[combination[1]-1]) - sympy.diff(functions[1], variables[combination[0]-1]) * sympy.diff(functions[0], variables[combination[1]-1]) )
            return bracket_result

#    def is_poisson_field(self, matrix, vector_field_coeff):
#        ''''''
#        dictionary_with_coeff = matrix_to_dictionary(matrix)
#        args_matrix = dictionary_with_coeff.pop('args')
#        variables = complete_args(args_matrix)
#        combinations = [x for x in itertools.combinations(range(1, len(variables)+1), 2)]

#       Schouten_W_P = 0
#        for combination in combinations:
#            Schouten_W_P_aux = 0
#            for k in range(len(variables)):
#                Schouten_W_P_aux = Schouten_W_P_aux + (vector_field_coeff[k] * sympy.diff(dictionary_with_coeff["{}{}".format(combination[0]-1, combination[1]-1)], variables[k]) - )

    def hamiltonian_vector_field(self, matrix, function):
        '''This method calculates a Hamiltonian vector field X_f, relative to Poisson bivector (matrix) π
        Parametres:
            * symbolic matrix π
            * string function f, this method converts it in symbolic variables
        return:
            X_f, that is a symbol variable
        '''
        # Convert matrix to dictionary
        dictionary_with_coeff = matrix_to_dictionary(matrix)
        # Get [x1, ..., xn] and [Dx1, ..., Dxn] sets
        variables, dx_basis = Dx_basis(matrix)
        # Obtains all the combinations from n in 2
        combinations = [x for x in itertools.combinations(range(1, len(variables)+1), 2)]
        # Convert the string function to symbolic function
        function = sympy.sympify(function)
        # Calculates the hamiltonian vector field
        hamiltonian_field = 0
        for combination in combinations:
            # Convert (i,j) to 'ij'
            ij_tuple_str = tuple(str(element) for element in combination)
            ij = ''.join(ij_tuple_str)
            # Calculates X_f(P) = Σ(π_{ij}[(∂f/∂x_i)Dxj - (∂f/∂x_j)Dxi])
            hamiltonian_field = hamiltonian_field + dictionary_with_coeff[ij] * ((sympy.diff(function, variables[combination[0]-1]) * dx_basis[combination[1]-1]) - (sympy.diff(function, variables[combination[1]-1]) * dx_basis[combination[0]-1]))

        return hamiltonian_field

    def pi_sharp_morphism(self, matrix, one_form_coeff):
        '''This method calculates the vector buldle morphism π^#(Ω)
        Parametres:
            * symbolic matrix π
            * list with coefficients from one_form Ω ordened the following form [coeff_1,...,coeff_n], such as,
            <[coeff_1,...,coeff_n],[dx1,...,dxn]> = coeff_1*dx1 + .... + coeff_n*dxn, also this method converts
            foer each element in a symbolic variables
        return:
            X_f, that is a symbol variable
        '''
        # Convert matrix to dictionary
        dictionary_with_coeff = matrix_to_dictionary(matrix)
        # Get [x1, ..., xn] and [dx1, ..., dxn] sets
        variables, dx_basis = Dx_basis(matrix, symbol='dx')
        # Obtains all the combinations from n in 2
        combinations = [x for x in itertools.combinations(range(1, len(variables)+1), 2)]
        # Calculates the morphism π^#(Ω)
        pi_sharp = 0
        for combination in combinations:
            # Convert (i,j) to 'ij'
            ij_tuple_str = tuple(str(element) for element in combination)
            ij = ''.join(ij_tuple_str)
            # Calculates π^#(Ω) = Σ(π_{ij}Ω_{i}dx_{j})
            pi_sharp = pi_sharp + dictionary_with_coeff[ij] * (sympy.sympify(one_form_coeff[combination[0]-1]) * dx_basis[combination[1]-1] - sympy.sympify(one_form_coeff[combination[1]-1]) * dx_basis[combination[0]-1])
        return pi_sharp

    def pi_sharp_in_kernel(self, pi_sharp_morphism):
        ''''''
        return True if pi_sharp_morphism == 0 else False

    def is_casimir(self, matrix, function):
        ''''''
        dictionary_with_coeff = matrix_to_dictionary(matrix)
        variables, dx_basis = Dx_basis(matrix)
        combinations = [x for x in itertools.combinations(range(1, len(variables)+1), 2)]
        function = sympy.sympify(function)

        casimir_sum = 0
        for combination in combinations:
            ij_tuple_str = tuple(str(element) for element in combination)
            ij = ''.join(ij_tuple_str)
            casimir_sum = casimir_sum + dictionary_with_coeff[ij] * sympy.diff(function, variables[combination[0]-1]) * dx_basis[combination[0]-1]

        return True if casimir_sum == 0 else False

###### LEGACY #########
    def create_symbolic_variables(self, dimension, variable='x'):
        ''' This method generates dynamic symbols variables from a
            number that is the dimension and the variable for the
            default is 'x'
        '''
        variables_list = []

        for i in range(dimension):
            # Define a string "x_1 wiht i = 1,..,dimension"
            x = "{}{}".format(variable, i + 1)
            # create symbols variables
            x = sympy.Symbol("{}".format(x))
            variables_list.append(x)

        # return a type symbolic list
        return variables_list


    def dx_basis(self, dimension):
        ''' This method creates a geometry con a orthonormal metric
            i.e; a basis {e_1,..., e_dimension} such that
            <e_{i}, e_{j}> = 1 if i=j &
            <e_{i}, e_{j}> = 0 if i!=j
        '''
        variables = self.create_symbolic_variables(dimension)
        g = ga(self.variable_string(dimension), g=self.list_with_one(dimension), coords=variables)
        # return a type multivector list g.mv()
        return g.mv()


    def is_poisson(self, dimension):
        '''This method calculating if a vector is of Poisson, through the SN bracket, i.e. [B,B]=0'''
        # Dynamic dictionary of symbolic variables
        variables = self.create_symbolic_variables(dimension)
        # Basis for cotangent space {dxi}_i=1,..n
        geometry = self.dx_basis(dimension)
        # Basis for the space {dxi^dxj}_i<j=1,..n
        basis = []
        i = 0
        while i < len(geometry):
            j = i + 1
            while j < len(geometry):
                basis.append(geometry[i] ^ geometry[j])
                j = j + 1
            i = i + 1

        # Calculing bivector
        bivector_list = []
        bivector = 0
        print("Please Only write the following variables {}".format(variables))
        for base in basis:
            str_variable_bivector = str(
                input("Enter a variable/constant for {}: ".format(base)))
            variable_bivector = sympy.sympify(str_variable_bivector)

            bivector_list.append(variable_bivector * base)

        for element in bivector_list:
            bivector = bivector + element
        print("You bivector are B = {}".format(bivector))

        # Get a list of components from bivector
        bi_components = bivector.components()

        # Calculing /frac{/partial bi}{partial xi}
        bi_diff_xi = {}
        for variable in variables:
            variable_list = []
            for element in bi_components:
                variable_list.append(element.diff(str(variable)))
            bi_diff_xi[variable] = variable_list

        # Calculing /frac{/partial bi}{partial dxi}
        bi_diff_dxi = {}
        for base in geometry:
            import code; code.interact(local=locals())
            base_str = str(base)
            variable_list = []
            for element in bi_components:
                element_str = str(element)
                if element_str.find(base_str) > 0:
                    factor = sympy.sympify(self.split_d(element_str)[0])
                    basis_aux = self.split(element_str)[1]
                    if basis_aux.find(base_str) == 0:
                        basis_aux = basis_aux.replace(base_str + '^', '')
                    else:
                        basis_aux = basis_aux.replace('^' + base_str, '')
                        factor = (-1) * factor

                    for base_aux in geometry:
                        base_aux_str = base_aux.raw_str()
                        if base_aux_str == basis_aux:
                            variable_list.append((factor * base_aux))
                else:
                    variable_list.append(0)
            bi_diff_dxi[base_str[1:]] = variable_list

        # Calculing [B, B]
        SN_bracket = 0
        for variable in variables:
            sum_xi = 0
            for xi_element in bi_diff_xi[variable]:
                sum_xi = sum_xi + xi_element
            print('d(B)/d{} = {}'.format(variable, sum_xi))
            sum_dxi = 0
            for dxi_element in bi_diff_dxi[variable]:
                sum_dxi = sum_dxi + dxi_element
            print('d(B)/d(d{}) = {}'.format(variable, sum_dxi))
            SN_bracket = SN_bracket + sum_xi ^ sum_dxi
            print('d(B)/d{}^d(B)/d(d{}) = {}'.format(variable,
                                                     variable, sum_xi ^ sum_dxi))

        # If SN_bracket is zero then the bivector is of Poisson
        if SN_bracket == 0:
            print("You bivector is Poisson")
        else:
            #Logic
            print(
                "You bivector is not Poisson, the SN bracket are: {}".format(SN_bracket))

        return SN_bracket
    
    def flaska_ratiu(self, dimension, return_bivector_coeff=False):
        '''This method calculates a Poisson bivector with Flaska-Ratiu method'''
        # Creates a symbolic variables
        variables_list = self.create_symbolic_variables(dimension)
        dx_list = self.dx_basis(dimension)

        # This block get to Casimir functions to user
        index = 1
        casimir_list = []
        print("Remember write the functions in terms of {}".format(variables_list))
        while index <= (dimension - 2):
            casimir_function = str(input("Enter to Camisir function: "))
            casimir_list.append(sympy.sympify(casimir_function))
            index = index + 1

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
        i = 1
        while i < len(diff_casimir_list):
            d_casimir = d_casimir ^ diff_casimir_list[i]
            i = i + 1

        if d_casimir.is_zero():
            return "Tu bivector es cero :("

        # This blocks obtains Poisson coefficients
        bivector_coeff_dict = {}
        combinations = [x for x in itertools.combinations(range(1, dimension + 1), 2)]
        for combination in combinations:
            combination = list(combination)
            casimir_matrix_without_combination = sympy.Matrix(diff_casimir_matrix)
            for i, element in enumerate(combination):
                casimir_matrix_without_combination.col_del(element - (i + 1))

            # Makes a dictionary with Poisson coefficients
            key = '{}{}'.format(combination[0],combination[1])
            coeff = ((-1)**(combination[0] + combination[1]))
            bivector_coeff_dict[key] = sympy.simplify(coeff * casimir_matrix_without_combination.det())

        # This method defines what option to return the user through de matrix_return variable
        if return_bivector_coeff:
            return bivector_coeff_dict

        # This block makes a Poisson bivector
        # Makes a basis {dxi^dxj}_{i<j} and it save in a dictionary type variable
        dx_ij_basis = {}
        i = 0
        while i < len(dx_list):
            j = i + 1
            while j < len(dx_list):
                key = "{}{}".format(i + 1, j + 1)
                dx_ij_basis[key] = dx_list[i] ^ dx_list[j]
                j = j + 1
            i = i + 1

        # Makes a Poisson Bivector
        if dx_ij_basis.keys() != bivector_coeff_dict.keys():
            return "Hubo un error al calcular el bivector lo siento :("

        bivector = 0
        estructure_symplectic_num = 0
        estructure_symplectic_dem = 0
        for key in dx_ij_basis.keys():
            bivector = bivector + bivector_coeff_dict[key] * dx_ij_basis[key]
            estructure_symplectic_num = estructure_symplectic_num + bivector_coeff_dict[key] * dx_ij_basis[key]
            estructure_symplectic_dem = estructure_symplectic_dem + bivector_coeff_dict[key] * bivector_coeff_dict[key]

        symplectic = estructure_symplectic_num * (-1 / estructure_symplectic_dem)
        # return a tuple with Poisson bivector and Symplectic form
        return (bivector, symplectic)

    def variable_string(self, dimension, symbol='dx'):
            '''This method generate dinamic variables
                from a number (dimension) in a string
            '''
            iteration = 1
            variable = ""

            while iteration <= dimension:
                # create symbols variables
                variable = variable + "{}{} ".format(symbol, iteration)
                iteration = iteration + 1
            # return a type symbolic list
            return variable[0:len(variable) - 1]

    def list_with_one(self, dimension):
        '''This method creates a "dimension" length
            list with value one in all the entries
        '''
        list_one = []
        for i in range(dimension):
            list_one.append(1)
        return list_one

    def split_d(self, string):
        ''' This method split a string when find to "d" in two strings
            note: a string is the form for example "(3x1-x2)*dx1^...^dxn"
        '''
        factor, basis = '', ''

        if 'd' in string:
            factor = string[:string.find('d') - 1]
            basis = string[string.find('d'):]

        return [factor, basis]

