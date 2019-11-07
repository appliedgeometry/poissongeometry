# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sympy as sym
import itertools as itools
from galgebra.ga import Ga


def two_tensor_form_to_matrix(dictionary, dim):
    """"""
    # Creates a symbolic Matrix
    bivector_matrix = sym.MatrixSymbol('B', dim, dim)
    bivector_matrix = sym.Matrix(bivector_matrix)

    # Assigns the corresponding coefficients of the two form
    for ij in itools.combinations_with_replacement(range(1, dim + 1), 2):
        i, j = ij[0], ij[1]
        # Makes the Poisson Matrix
        bivector_matrix[i - 1, j - 1] = 0 if i == j else sym.sympify(dictionary[(i,j)])
        bivector_matrix[j - 1, i - 1] = (-1) * bivector_matrix[i - 1, j - 1]

    # Return a symbolic Poisson matrix or Poisson matrix in LaTeX syntax
    return bivector_matrix


def symbolic_expression(dictionary, dim, coordinates, variable, dx=False):
    """This method converts a dictionary with symbolic values ​​in a symbolic expression
    """

    variables = variable_string(dim, symbol=F'D{variable}')
    if dx:
        variables = variable_string(dim, symbol=F'd{variable}')

    basis = Ga(variables, g=sym.eye(dim), coords=coordinates)
    # return a type multivector list g.mv()
    basis = basis.mv()

    symbolic_expresion = 0
    for index in dictionary.keys():
        if type(index) == int:
            symbolic_expresion = symbolic_expresion + basis[index-1] * dictionary[index]
        else:
            element_basis = basis[index[0]-1]
            for i in index:
                element_basis = element_basis ^ basis[i-1] if i != index[0] else element_basis
            symbolic_expresion = symbolic_expresion + element_basis * dictionary[index]

    return symbolic_expresion


def variable_string(dimension, symbol):
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


def show_coordinates(coordinates):
    ''''''
    if len(coordinates) > 2:
        return f'({coordinates[0]},...,{coordinates[-1]})'
    else:
        return f'Dimension do not sufficient'


def basis(dim, coordinates, variable, dx=False):
    """ """

    variables = variable_string(dim, symbol=F'D{variable}')
    if dx:
        variables = variable_string(dim, symbol=F'd{variable}')

    basis = Ga(variables, g=sym.eye(dim), coords=coordinates)
    # return a type multivector list g.mv()
    return basis.mv()


def symbolic_expression_2(dict, basis):
    """This method converts a dictionary with symbolic values ​​in a symbolic expression
    """
    if len(list(dict.keys())) == 1 and list(dict.keys())[0] == 0:
        symbolic_expresion = dict[list(dict.keys())[0]]
    else:
        symbolic_expresion = 0
        for index in dict.keys():
            element_basis = basis[int(str(index)[0])-1]
            for i in str(index):
                element_basis = element_basis ^ basis[int(i)-1] if i != str(index)[0] else element_basis
            symbolic_expresion = symbolic_expresion + element_basis * dict[index]

    return symbolic_expresion


def derivate_formal(bivector, index):
    """"""
    bivector = {0: sym.sympify(bivector)} if isinstance(bivector, str) else bivector
    derivate_formal = {}
    for bivector_key in bivector.keys():
        len_key = len(bivector_key)
        for position, value in enumerate(bivector_key):
            if str(index) == value:
                if len(bivector_key) == 1:
                    derivate_formal.update({(): bivector[bivector_key]})
                else:
                    factor = -1 if (len_key - position) % 2 == 0 else 1
                    derivate_formal_key = bivector_key.copy()
                    derivate_formal.update({tuple(list(derivate_formal_key).remove(value)): factor * bivector[bivector_key]})
    return derivate_formal


def derivate_standar(bivector, variable):
    """"""
    bivector = {0: sym.sympify(bivector)} if isinstance(bivector, str) else bivector
    derivate_standar = {}
    for bivector_key in bivector.keys():
        derivate_standar.update({bivector_key: bivector[bivector_key].diff(variable)})
    return derivate_standar


def symbolic_multivector_to_dict(multivector_symbolic, manifold_dim):
    """"""
    # This block creates the keys in string type
    keys_str = ['0']
    i = 1
    while i < manifold_dim + 1:
        for z in itools.combinations(range(1, manifold_dim+1), i):
            keys_str.append("".join(map(str, z)))
        i = i + 1

    # Here obtains all coeff from a multivector
    multivector_sym_coefs = multivector_symbolic.blade_coefs()
    multivector_raw_dict = dict(zip(keys_str, multivector_sym_coefs))
    return multivector_raw_dict


def values_to_kernel(multivector, variables, dimension, coordinates):
    """"""
    m = {}
    i = 1
    while i < dimension+1:
        m.update({str(coordinates[i-1]): 0})
        i = i + 1

    equations = []
    n = m.copy()
    for x in coordinates:
        n.pop(str(x))
        for key, value in multivector.items():
            equations.append(value.subs(n))
        n = m.copy()

    values = sym.solve(equations, variables)
    return values

def remove_values_zero(dictionary):
    """"""
    if all(value == 0 for value in dictionary.values()):
        return {0: 0}
    else:
        clean_dict = {}
        for key, value in dictionary.items():
            if value != 0:
                clean_dict.update({key: value})
        return clean_dict


class DimensionError(Exception):
    """Clase base para excepciones en el módulo."""
    pass


def validate_dimension(dimension):
    """This method check if the dimension variable is valid for the this class"""
    if not isinstance(dimension, int):
        raise TypeError(F"Your variable is not a int type")

    if dimension < 2:
        raise DimensionError(F"Your dimension is not greater than two")
    else:
        return dimension


def is_dicctionary(dicctionary):
    """This method check if the dimension variable is valid for the this class"""
    if not isinstance(dicctionary, dict):
        raise TypeError(F"Your variable is not a dictionary type")

    return dicctionary


def is_string(dicctionary):
    """This method check if the dimension variable is valid for the this class"""
    if not isinstance(dicctionary, str):
        raise TypeError(F"Your variable is not a dictionary type")

    return dicctionary
