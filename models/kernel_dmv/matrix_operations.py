#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy 

def get_position_of_minimum(matrix):
    return numpy.unravel_index(numpy.nanargmin(matrix), matrix.shape)

def get_position_of_maximum(matrix):
    return numpy.unravel_index(numpy.nanargmax(matrix), matrix.shape)

def get_distance_matrix(cell_grid_x, cell_grid_y, x, y):
    return numpy.sqrt((x - cell_grid_x) ** 2 + (y - cell_grid_y) ** 2)

def get_distance_matrix_squared(cell_grid_x, cell_grid_y, x, y):
    return (x - cell_grid_x) ** 2 + (y - cell_grid_y) ** 2