# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from galgebra.ga import Ga as ga
import sympy
import itertools


def matrix_to_dictionary(matrix):
    '''
    This method receives a Poisson Matrix, i.e. a symbols antisymmetric matrix with zero diagonal and
    we get the values to obtains the Poisson bivector associate to matrix
    '''
    # Create a dictionary with π_ij values of bivector
    try:
        dimension_bivector = sympy.sqrt(len(matrix))
    except:
        print("The matrix is ​​not square")
        return {}
    
    combinations = [x for x in itertools.combinations(range(1, dimension_bivector + 1), 2)]
    dictionary_with_coeff = {}
    i = 0
    while i < dimension_bivector + 1:
        j = i + 1
        while j < dimension_bivector + 1:
            if (i,j) in combinations:
                dictionary_with_coeff.update({'{}{}'.format(i,j): matrix[i-1,j-1]})
            j = j + 1
        i = i + 1

    # add args the Poisson matrix (dictionary type)
    dictionary_with_coeff['args'] = matrix.free_symbols

    return dictionary_with_coeff

def complete_args(args_set):
    '''
    '''
    # This block obatins index and variable from args_set
    indexes = []
    variable = []
    for args in args_set:
        index = ''
        for position in str(args):
            if position.isdigit():
                index = index + position
            else:
                variable.append(position)
        indexes.append(int(index))
    # get the indexes in order
    indexes = sorted(indexes)

    if len(set(variable)) > 1:
        print('No tienes variables con indices')

        return "ERROR"

    args = sympy.symbols('{}1:{}'.format(variable[0], (indexes[len(indexes)-1] + 1)))

    # return symbolic tuple
    return args

def Dx_basis(matrix, symbol='Dx'):
    ''''''
    dictionary_with_coeff = matrix_to_dictionary(matrix)
    args_matrix = dictionary_with_coeff.pop('args')
    variables = complete_args(args_matrix)
    #import code; code.interact(local=locals())
    dimension = sympy.sqrt(len(matrix))
    Dx = ga(variable_string(dimension, symbol=symbol), g=list_with_one(dimension), coords=variables)
    return variables, Dx.mv()

def variable_string(dimension, symbol='dx'):
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

def list_with_one(dimension):
    '''This method creates a "dimension" length
        list with value one in all the entries
    '''
    list_one = []
    for i in range(dimension):
        list_one.append(1)
    return list_one
