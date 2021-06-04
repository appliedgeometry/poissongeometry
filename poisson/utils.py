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

    This file contains the all auxiliary functions to use in the Poisson geometry module
"""
from __future__ import unicode_literals
from poisson.errors import DimensionError


def validate_dimension(dim):
    """ This method check if the dimension variable is valid for the this class"""
    if not isinstance(dim, int):
        raise DimensionError(F"{dim} is not int")

    if dim < 2:
        raise DimensionError(F"{dim} < 2")
    else:
        return dim


def del_columns(matrix, columns):
    """"""
    copy_matrix = matrix.copy()
    copy_matrix.col_del(columns[0])
    copy_matrix.col_del(columns[1] - 1)
    return copy_matrix


def show_coordinates(coordinates):
    """"""
    if len(coordinates) >= 4:
        return f'({coordinates[0]},...,{coordinates[-1]})'
    else:
        return f'({coordinates})'
