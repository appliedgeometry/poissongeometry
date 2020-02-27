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
import itertools as itools
from galgebra.ga import Ga


def two_tensor_form_to_matrix(dictionary, dim):
    """ Constructs the matrix of a bivector field."""
    # Creates a symbolic Matrix
    bivector_matrix = sym.MatrixSymbol('B', dim, dim)
    bivector_matrix = sym.Matrix(bivector_matrix)

    # Assigns the corresponding coefficients of the two form
    for ij in itools.combinations_with_replacement(range(1, dim + 1), 2):
        i, j = ij[0], ij[1]
        # Makes the Poisson Matrix
        bivector_matrix[i - 1, j - 1] = 0 if i == j else sym.sympify(dictionary[(i, j)])
        bivector_matrix[j - 1, i - 1] = (-1) * bivector_matrix[i - 1, j - 1]

    # Return a symbolic Poisson matrix or Poisson matrix in LaTeX syntax
    return bivector_matrix


def symbolic_expression(dictionary, dim, coordinates, variable, dx=False):
    """ This method converts a dictionary with symbolic values ​​in a symbolic expression"""

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
    """ This method generate dinamic variables from a number (dimension) in a string"""
    iteration = 1
    variable = ""

    while iteration <= dimension:
        # create symbols variables
        variable = variable + "{}{} ".format(symbol, iteration)
        iteration = iteration + 1
    # return a type symbolic list
    return variable[0:len(variable) - 1]


def show_coordinates(coordinates):
    """ This method shows the current coordinates from Poisson Geometry Class"""
    if len(coordinates) > 2:
        return f'({coordinates[0]},...,{coordinates[-1]})'
    else:
        return f'Dimension do not sufficient'


def basis(dim, coordinates, variable, dx=False):
    """ This method generates a basis for tangent space or your dual, i.e., is a basis of type {dx_i} or {Dx_i}"""

    variables = variable_string(dim, symbol=F'D{variable}')
    if dx:
        variables = variable_string(dim, symbol=F'd{variable}')

    basis = Ga(variables, g=sym.eye(dim), coords=coordinates)
    # return a type multivector list g.mv()
    return basis.mv()


def remove_values_zero(dictionary):
    """ This method updates a dictionary removing all values that are zero"""
    if all(value == 0 for value in dictionary.values()):
        return {0: 0}
    else:
        clean_dict = {}
        for key, value in dictionary.items():
            if value != 0:
                clean_dict.update({key: value})
        return clean_dict


class DimensionError(Exception):
    pass


def validate_dimension(dimension):
    """ This method check if the dimension variable is valid for the this class"""
    if not isinstance(dimension, int):
        raise TypeError(F"Your variable is not a int type")

    if dimension < 2:
        raise DimensionError(F"Your dimension is not greater than two")
    else:
        return dimension
