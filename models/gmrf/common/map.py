import numpy as np
from .lattice import Lattice, LatticeScalar, LatticeScalar, LatticeVector
from abc import abstractmethod
import math


#   Visual representation of coordinate systems (x,y) vs (i,j)
#
#       -----> j
#       |	+------------------+
#       |	|                  |
#       V	|                  |
#       i	|                  |
#           |     LATTICE      |
#           |                  |
#       y	|                  |
#       ^	|                  |
#       |	|                  |
#       |	+------------------+
#       -----> x
#


##==============================================================================
class Map:

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, offset=0):
        assert type(dimensions) == int
        assert dimensions > 0
        assert type(size) == tuple
        assert len(size) == dimensions
        for width in size:
            assert width > 0

        if offset == 0:
            offset = tuple(np.zeros(dimensions).tolist())
        else:
            assert len(offset) == dimensions

        ## Member variables
        self.size = size
        self.dimensions = dimensions
        self.offset = offset

        return

    ## Abstract ----------------------------------------------------------------

    @abstractmethod
    def plot(self):
        pass


##============================================================================
class DiscreteMap(Map, Lattice):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution=1.0, offset=0):

        ## Parameter check
        assert type(dimensions) == int, type(dimensions)
        assert dimensions > 0
        assert resolution > 0.0
        assert type(size) == tuple, str(size)

        ## Compute aux
        shape = np.zeros(dimensions)
        for axis in range(dimensions):
            if axis == 0:
                x = size[0]
                j = int(math.ceil(x / resolution))
                shape[1] = j
            elif axis == 1:
                y = size[1]
                i = int(math.ceil(y / resolution))
                shape[0] = i
            else:
                num_cells = np.round(size[axis] / resolution).astype(int)
                shape[axis] = num_cells
        shape = tuple(shape.astype(int).tolist())
        assert len(shape) == dimensions
        assert type(shape[0]) == int

        ## Init base
        Map.__init__(self, dimensions=dimensions, size=size, offset=offset)
        Lattice.__init__(self, dimensions=dimensions, shape=shape)

        ## Member variables
        self.shape = shape
        self.resolution = resolution
        self.dimensions = dimensions

        ## Quick check for 2D
        if self.dimensions >= 2:
            assert np.isclose(self.size[0], self.shape[1] * self.resolution, self.resolution / 2), \
                str(self.size[0]) + " != " + str(self.shape[1]) + " * " + str(self.resolution)
            assert np.isclose(self.size[1], self.shape[0] * self.resolution, self.resolution / 2), \
                str(self.size[1]) + " != " + str(self.shape[0]) + " * " + str(self.resolution)

        return

    ## METHODS -----------------------------------------------------------------

    def getPosition(self, position, format='ij'):
        assert self.isPositionValid(position)
        cell = self._convertPositionToCell(position)
        value = self.getCell(cell)
        if format == 'ij':
            return value
        elif format == 'xy':
            x = value[1]
            y = -value[0]
            value_xy = list(value)
            value_xy[0] = x
            value_xy[1] = y
            return tuple(value_xy)
        else:
            assert False


    def setPosition(self, position, value):
        assert self.isPositionValid(position)
        cell = self._convertPositionToCell(position)
        return self.setCell(cell, value)

    def getStraightPathBetweenPositions(self, start_position, end_position, step_size=-1):
        assert self.isPositionValid(start_position)
        assert self.isPositionValid(end_position)
        start =  np.array(start_position)
        end =  np.array(end_position)

        if step_size < 0:
            step_size = self.resolution / 2

        ## Compute movement parameters
        vector = end - start
        vector_length = np.linalg.norm(vector)
        if vector_length < step_size:
            return [start_position]

        unity_vector = vector / (vector_length + 1e-10)
        step_vector = step_size * unity_vector
        num_steps = int(np.floor(vector_length / step_size))

        ## Iterate in a straight line
        path = []
        for steps in range(num_steps):
            movement_vector = steps * step_vector
            new_position = start + np.array(movement_vector)
            new_position = tuple(new_position.tolist())
            path += [new_position]

        ## Add last position to make sure we pass through it
        path += [end_position]
        assert self.isPositionValid(path)
        return path

    ## PRIVATE METHODS ---------------------------------------------------------

    def isPositionValid(self, position):
        if type(position) is tuple:
            assert len(position) == self.dimensions
            for axis in range(0, len(position)):
                x = position[axis]
                max_axis = self.size[axis] - self.offset[axis]
                min_axis = 0 - self.offset[axis]
                assert min_axis <= x <= max_axis, \
                    "Axis " + str(axis) + ": " + str(min_axis) + "<=" + str(x) + "<=" + str(max_axis)

            ## Quick check for 2D
            if self.dimensions >= 2:
                assert np.isclose(self.size[0], self.shape[1] * self.resolution, self.resolution / 2), \
                    str(self.size[0]) + " != " + str(self.shape[1]) + " * " + str(self.resolution)
                assert np.isclose(self.size[1], self.shape[0] * self.resolution, self.resolution / 2), \
                    str(self.size[1]) + " != " + str(self.shape[0]) + " * " + str(self.resolution)

        elif type(position) is list:
            for p in position:
                assert self.isPositionValid(p)

        return self

    def _convertPositionToCell(self, position, fix_position=False):
        if fix_position:
            position = self._clipPosition(position)
        else:
            assert self.isPositionValid(position)

        if type(position) is tuple:
            position = np.array(position) + np.array(self.offset)

            ## Adjust if at the very border: Move inside map
            fixed_position = np.zeros(self.dimensions)
            for axis in range(self.dimensions):
                x = position[axis]
                ok_error = self.resolution / 10
                min = 0 + ok_error
                max = self.size[axis] - ok_error
                if x > max:
                    x = max
                elif x < min:
                    x = min
                fixed_position[axis] = x
            position = tuple(fixed_position.tolist())

            cell = np.zeros(len(position), dtype=int)
            for axis in range(0, len(position)):
                if len(position) >= 2 and axis == 0:  # x -> j
                    x = position[axis]
                    j = int(np.floor(x / self.resolution))
                    cell[1] = j
                elif len(position) >= 2 and axis == 1:  # y -> -i
                    y = position[axis]
                    max_y = self.size[1]
                    i = int(np.floor((max_y - y) / self.resolution))
                    cell[0] = i
                else:
                    distance = position[axis]
                    cell_index = int(np.floor(distance / self.resolution))
                    cell[axis] = cell_index

            ## Fix if at the very border of the map
            for axis in range(0, len(cell)):
                index = cell[axis]
                if index == max:
                    cell[axis] = max - 1

            ## Convert to tuple
            cell = tuple(cell.tolist())

        elif type(position) is list:
            cell = [self._convertPositionToCell(p) for p in position]
        else:
            assert False

        assert self._isCellValid(cell)
        return cell

    def _convertCellToPosition(self, cell):
        assert self._isCellValid(cell)

        if type(cell) is tuple:
            assert len(cell) == self.dimensions

            position = np.zeros(self.dimensions)
            for axis in range(0, self.dimensions):

                if self.dimensions >= 2 and axis == 0:  # i -> y
                    i = cell[axis]
                    max_y = self.size[1] - 1e-12
                    y = y = (max_y - (i + 0.5) * self.resolution)
                    position[1] = y
                elif self.dimensions >= 2 and axis == 1:  # j -> x
                    j = cell[axis]
                    x = (j + 0.5) * self.resolution
                    position[0] = x
                else:
                    index = cell[axis]
                    distance = (index + 0.5) * self.resolution
                    position[axis] = distance

            ## Convert to tuple, then clip
            position = np.round(position - np.array(self.offset), 6)
            position = tuple(position.tolist())
            position = self._clipPosition(position)

        elif type(cell) is list:
            position = [self._convertCellToPosition(c) for c in cell]

        else:
            return False

        assert self.isPositionValid(position)
        return position

    def _clipPosition(self, position):
        if type(position) is tuple:
            assert len(position) == self.dimensions
            clipped_position = ()
            for axis in range(0, self.dimensions):
                x = position[axis]
                min_x = 0 - self.offset[axis]
                max_x = self.size[axis] - self.offset[axis]
                clipped_x = max(min(x, max_x), min_x)
                clipped_position += (clipped_x,)

        elif type(position) is list:
            clipped_position = [self._clipPosition(p) for p in position]

        else:
            assert False

        assert self.isPositionValid(clipped_position)
        return clipped_position

    def _convertPositionToIndex(self, position):
        assert self.isPositionValid(position)

        if type(position) is tuple:
            cell = self._convertPositionToCell(position)
            index = self._convertCellToIndex(cell)
            return index

        elif type(position) is list:
            return [self._convertPositionToIndex(p) for p in position]

        else:
            assert False

    ## METHODS -----------------------------------------------------------------

    def getMaxPosition(self):
        cell = self.getMaxCell()
        return self._convertCellToPosition(cell)


##============================================================================

class DiscreteScalarMap(LatticeScalar, DiscreteMap):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution=1.0, init_value=0, offset=0):
        ## Init base
        assert type(size) == tuple
        assert dimensions == len(size)
        DiscreteMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution, offset=offset)
        LatticeScalar.__init__(self, dimensions=dimensions, shape=self.shape, init_value=init_value)
        return

    @classmethod
    def fromPGM(cls, pgm_file, resolution=1, byteorder='>'):
        lattice = LatticeScalar.fromPGM(pgm_file, byteorder)
        instance = cls.fromMatrix(lattice.toMatrix(), resolution)
        assert instance.dimensions == 2
        return instance

    @classmethod
    def fromMatrix(cls, matrix, resolution=1.0, offset=0):
        assert resolution > 0
        dimensions = len(matrix.shape)
        size = np.zeros(dimensions)
        for axis in range(dimensions):

            assert matrix.shape[axis] > 0
            if dimensions >= 2 and axis == 0:  # i -> y
                max_i = matrix.shape[axis]
                max_y = max_i * resolution
                size[1] = max_y
            elif dimensions >= 2 and axis == 1:  # j -> x
                max_j = matrix.shape[axis]
                max_x = max_j * resolution
                size[0] = max_x
            else:
                max_index = matrix.shape[axis]
                max_distance = max_index * resolution
                size[axis] = max_distance

        size = tuple(size.tolist())
        instance = cls(dimensions=dimensions, size=size, resolution=resolution, offset=offset)
        instance.loadMatrix(matrix)
        return instance

    def __getitem__(self, key):
        return DiscreteScalarMap.fromMatrix(self._data[key], resolution=self.resolution)


##============================================================================
class DiscreteVectorMap(LatticeVector, DiscreteMap):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution, init_value=0, offset=0):
        ## Parameter check
        assert type(size) == tuple, str(type(size))
        assert dimensions == len(size)

        ## Init base
        DiscreteMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution, offset=offset)
        LatticeVector.__init__(self, dimensions=dimensions, shape=self.shape, init_value=init_value)

        return

    @classmethod
    def fromMatrix(cls, matrix_i, matrix_j, resolution=1.0):
        instance = DiscreteVectorMap((matrix_i, matrix_j), resolution)
        assert instance.dimensions == 2
        return instance

    @classmethod
    def fromMatrix(cls, matrix_tuple, resolution=1.0, offset=0):

        assert type(matrix_tuple) == tuple
        dimensions = len(matrix_tuple)
        shape = matrix_tuple[0].shape
        for matrix in matrix_tuple:
            assert shape == matrix.shape

        matrix = matrix_tuple[0]
        size = np.zeros(dimensions)
        for axis in range(0, dimensions):
            assert matrix.shape[axis] > 0
            if dimensions >= 2 and axis == 0:  # i -> y
                max_i = matrix.shape[axis]
                max_y = max_i * resolution
                size[1] = max_y
            elif dimensions >= 2 and axis == 1:  # j -> x
                max_j = matrix.shape[axis]
                max_x = max_j * resolution
                size[0] = max_x
            else:
                max_index = matrix.shape[axis]
                max_distance = max_index * resolution
                size[axis] = max_distance

        size = tuple(size.tolist())
        instance = cls(dimensions=dimensions, size=size, resolution=resolution, offset=offset)
        instance.loadMatrix(matrix_tuple)
        return instance

    def __getitem__(self, key):
        assert self.dimensions == 2
        wind_i = self._data[...,0][key]
        wind_j = self._data[...,1][key]
        return DiscreteVectorMap.fromMatrix((wind_i, wind_j), resolution=self.resolution)
