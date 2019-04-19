# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ga
import sympy
import itertools


class PoissonGeometry(object):

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
        g = ga.Ga(self.variable_string(dimension), g=self.list_with_one(dimension), coords=variables)
        # return a type multivector list g.mv()
        return g.mv()


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
            casimir_function = str(raw_input("Enter to Camisir function: "))
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
