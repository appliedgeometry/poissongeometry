# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sympy as sym
from galgebra.ga import Ga


def symbolic_expression(dict, dim, coordinates, variable, dx=False):
    """This method converts a dictionary with symbolic values ​​in a symbolic expression
    """

    variables = variable_string(dim, symbol=F'D{variable}')
    if dx:
        variables = variable_string(dim, symbol=F'd{variable}')

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

def show_coordinates(coordinates):
    ''''''
    if len(coordinates) > 2:
        return f'({self.coordinates[0]},...,{self.coordinates[-1]})'
    else:
        return f'Dimension do not sufficient'


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
