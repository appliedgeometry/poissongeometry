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

    This file contains the Errors Custom class to
    use in the module Poisson geometry module
"""


class DimensionError(Exception):
    pass


class MultivectorError(Exception):
    pass


class FunctionError(Exception):
    pass


class DiferentialFormError(Exception):
    pass


class CasimirError(Exception):
    pass


class Nonlinear(Exception):
    pass


class Nonhomogeneous(Exception):
    pass
