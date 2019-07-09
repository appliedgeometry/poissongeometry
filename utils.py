# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from galgebra.ga import Ga
from errors import DimensionError

def dict_to_symbol_exp(dict, dimension, coordinates, dx=False):
    """This method converts a dictionary with symbolic values ​​in a symbolic expression
    """

    variables = variable_string(dim, symbol='Dx')
    if dx:
        variables = variable_string(dim, symbol='dx')

    basis = Ga(variables, g=sym.eye(dim), coords=coordinates)
    # return a type multivector list g.mv()
    basis = basis.mv()

    symbolic_expresion = 0
    for index in dict.keys():
        element_basis = basis[int(str(index)[0])-1]
        for i in str(index):
            element_basis = element_basis ^ basis[int(i)-1] if i != str(index)[0] else element_basis
        symbolic_expresion = symbolic_expresion + element_basis * dict[index]

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

def validate_dimension(dimension):
    """This method check if the dimension variable is valid for the this class"""
    if not isinstance(dimension, int):
        raise TypeError(F"Your variable is not a type int")

    if dimension < 2:
        raise DimensionError(F"Your dimension is not greater than two")
    else:
        return dimension
