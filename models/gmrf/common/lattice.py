import copy
import math
import re
from abc import abstractmethod

import numpy as np

from models.gmrf.utils.report.plot import plotScalarField, plotVectorField, plotScalar3DField


##==============================================================================
class Lattice:

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, shape, dtype=float):
        assert type(dimensions) == int, type(dimensions)
        assert dimensions > 0
        assert type(shape) == tuple
        assert len(shape) == dimensions, str(len(shape)) + " != " + str(dimensions)
        for x in shape:
            assert x >= 0

        ## Get number of cells
        num_cells = 1
        for x in shape:
            num_cells *= x

        ## Member variables
        self.shape = shape
        self.dimensions = dimensions
        self._num_cells = num_cells
        self._data = np.zeros(self.shape, dtype=dtype)

        return

    @classmethod
    def fromMatrix(cls, matrix):
        if type(matrix) == np.ndarray:
            shape = matrix.shape
            dimensions = len(shape)
            instance = cls(dimensions=dimensions, shape=shape)
            instance.loadMatrix(matrix)
            return instance
        elif type(matrix) == tuple:
            matrix_tuple = matrix
            dimensions = len(matrix_tuple)
            shape = matrix_tuple[0].shape
            for matrix in matrix_tuple:
                assert shape == matrix.shape
            instance = cls(dimensions=dimensions, shape=shape)
            instance.loadMatrix(matrix_tuple)
            return instance
        else:
            assert False

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getCell(self, cell):
        data = self._data[cell]
        if type(data) == np.ndarray:
            data = tuple(data.tolist())
        return data

    @abstractmethod
    def _setCell(self, cell, value):
        self._data[cell] = value
        return self

    @abstractmethod
    def _plot(self, vmax, mask, save, z, interpol, scale):
        pass

    @abstractmethod
    def _normalize(self):
        pass

    @abstractmethod
    def _getMaxCell(self):
        pass

    @abstractmethod
    def _fill(self, value):
        self._data[:] = value
        return self

    ## METHODS -----------------------------------------------------------------

    def getCell(self, cell):
        assert self._isCellValid(cell)

        if type(cell) is tuple:
            return self._getCell(cell)
        elif type(cell) is list:
            return [self.getCell(c) for c in cell]
        else:
            assert False, str(type(cell)) + " is not supported by this funcion"

    def setCell(self, cell, value):
        assert self._isCellValid(cell)
        assert self.__isValueValid(value)

        if type(cell) is tuple:
            self._setCell(cell, value)

        elif type(cell) is list and type(value) is list:
            assert len(cell) == len(value)
            [self._setCell(cell[i], value[i]) for i in range(0, len(cell))]

        elif type(cell) is list and type(value) is not list:
            [self._setCell(c, value) for c in cell]

        else:
            assert False

        return self

    def getMaxCell(self):
        cell = self._getMaxCell()
        self._isCellValid(cell)
        return cell

    def normalize(self):
        return self._normalize()

    def plot(self, vmax=0, mask=0, save="", z='', interpol=3, scale=15):
        return self._plot(vmax=vmax, mask=mask, save=save, z=z, interpol=interpol, scale=scale)

    def fill(self, value):
        assert type(value) != list
        assert self.__isValueValid(value)
        return self._fill(value)

    def toMatrix(self):
        if self.__isScalarType():
            assert type(self._data is np.ndarray)
            matrix = copy.deepcopy(self._data)
            assert matrix.shape == self.shape
            return matrix
        elif self.__isVectorType():
            matrices = [copy.deepcopy(self._data[..., i]) for i in range(self.__getDataDimensions())]
            assert len(matrices) == self.__getDataDimensions(), str(len(matrices)) + " != " + str(
                self.__getDataDimensions())
            return tuple(matrices)
        else:
            assert False

    def loadMatrix(self, matrix):
        if self.__isScalarType():
            assert (self.shape == matrix.shape), str(self.shape) + " != " + str(matrix.shape)
            assert type(matrix) == np.ndarray
            # assert matrix.dtype == self._data.dtype, str(matrix.dtype) + " != " + str(self._data.dtype)
            self._data = matrix

        elif self.__isVectorType():
            assert type(matrix) == tuple
            assert len(matrix) == self.dimensions
            shape = matrix[0].shape

            for axis in range(self.dimensions):
                m = matrix[axis]
                assert m.dtype == self._data.dtype
                assert m.shape == shape
                self._data[..., axis] = m
        else:
            assert False
        return self

    ## PRIVATE -----------------------------------------------------------------

    def _getAllCellCoordinatesBetween(self, start_cell, end_cell):
        assert self._isCellValid(start_cell)
        assert self._isCellValid(end_cell)

        start = np.array(start_cell)
        end = np.array(end_cell)

        vector = end - start
        vector_length = np.linalg.norm(vector)
        if vector_length == 0:
            return [start_cell]

        unity_vector = np.array(vector / (vector_length + 1e-10))
        steps = np.arange(np.ceil(vector_length + 1))
        intermediate_cells = \
            list(map(tuple, ((steps * unity_vector[:, np.newaxis] + start[:, np.newaxis]).astype(int).T.tolist())))

        intermediate_cells += [end_cell]
        cells = list(dict.fromkeys(intermediate_cells))  # Keep unique cells only
        assert cells[0] == start_cell
        assert cells[-1] == end_cell
        assert self._isCellValid(cells)
        return cells

    ## Ensure cell indices fall within lattice
    def _clipCell(self, cell):
        if type(cell) == tuple:
            clipped_cell = cell
            for x in range(0, len(cell)):
                i = cell[x]
                max_i = self.shape[x]
                min_i = 0
                clipped_i = max(min(i, max_i), min_i)
                clipped_cell[x] = clipped_i

        elif type(cell) == list and type(cell[0] == tuple):
            clipped_cell = [self._clipCell(c) for c in cell]
        else:
            assert False

        self._isCellValid(clipped_cell)
        return clipped_cell

    def _convertCellToIndex(self, cell):
        assert self._isCellValid(cell)

        if type(cell) is tuple:
            ## For 2D, with axis (i,j), this should be equal to:
            ## index = i + j*max_i
            ## For 3D, with (i,j,k), it should be:
            ## index = i + j*max_i + k*max_i*max_j
            multiplier = 1
            index = 0
            for x in range(self.dimensions):
                i = cell[x]
                index += i * multiplier
                max_axis = self.shape[x]
                multiplier *= max_axis

        elif type(cell) is np.ndarray:
            assert False  # Not implemente yet
        elif type(cell) is list:
            index = [self._convertCellToIndex(c) for c in cell]
        else:
            assert False

        assert self.__isIndexValid(index)
        return index

    def _convertIndexToCell(self, index):

        if type(index) is int:
            cell_inverse = []
            remainder = index

            for axis in range(self.dimensions - 1, 0, -1):
                multiplier = 1
                for prev_axis in range(axis - 1, -1, -1):
                    multiplier *= self.shape[prev_axis]

                i = math.floor(remainder / multiplier)
                remainder = remainder % multiplier
                cell_inverse += [i]
            cell_inverse += [remainder]

            cell_inverse.reverse()
            cell = tuple(np.array(cell_inverse).tolist())  # ugly hack

        elif type(index) == list:
            cell = [self._convertIndexToCell(i) for i in index]

        elif type(index) == np.ndarray:
            return self._convertIndexToCell(index.tolist())

        else:
            assert False, type(index)

        assert self._isCellValid(cell)
        return cell

    def _isCellValid(self, cell):

        if type(cell) is tuple:
            assert type(cell) is tuple
            assert len(cell) == self.dimensions
            for x in cell:
                assert type(x) == int, type(x)

            for x in range(0, len(cell)):
                total_cells_axis = self.shape[x]
                i = cell[x]
                assert i >= 0, \
                    "Cell axis '" + str(x) + "' = " + str(i) + " is smaller than zero"
                assert i < total_cells_axis, \
                    "Cell axis '" + str(x) + "' = " + str(i) + \
                    " is too big and falls out of the lattice in that axis (max = " + str(total_cells_axis) + \
                    ", size=" + str(self.shape) + ", shape=" + str(self.shape) + ")"

        elif type(cell) is list:
            [self._isCellValid(c) for c in cell]

        return True

    ## PRIVATE -----------------------------------------------------------------

    def __isIndexValid(self, index):
        if type(index) == int:
            assert (0 <= index <= self._num_cells)
        elif type(index) == list:
            [self.__isIndexValid(i) for i in index]
        else:
            assert False

        return True

    def __getDataDimensions(self):
        if self.dimensions == len(self._data.shape):
            return 1
        elif self.dimensions + 1 == len(self._data.shape):
            return self._data.shape[-1]
        else:
            assert False

    def __isScalarType(self):
        return self.__getDataDimensions() == 1

    def __isVectorType(self):
        return self.__getDataDimensions() > 1

    def __isValueValid(self, value):
        if type(value) == list:
            [self.__isValueValid(v) for v in value]
        elif type(self._data.dtype) == tuple:
            assert type(value) == tuple
            assert len(value) == len(self._data.dtype), "Dimension of value not compatible with lattice"
        elif type(self._data.dtype) == type:
            return True
        return True

    def getCellPattern_1x1(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]

        i = np.arange(cells_i)
        j = np.arange(cells_j)
        cells = list(map(tuple, np.array(np.meshgrid(i, j, indexing='ij')).T.reshape(-1, 2).tolist()))

        #assert self._isCellValid(cells)
        return cells

    def getCellPattern_1x2(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]

        i_l = np.arange(0, cells_i)
        j_l = np.arange(0, cells_j - 1)
        cells_l = list(map(tuple, np.array(np.meshgrid(i_l, j_l, indexing='ij')).T.reshape(-1, 2).tolist()))

        i_r = np.arange(0, cells_i)
        j_r = np.arange(1, cells_j)
        cells_r = list(map(tuple, np.array(np.meshgrid(i_r, j_r, indexing='ij')).T.reshape(-1, 2).tolist()))

        #assert self._isCellValid(cells_l)
        #assert self._isCellValid(cells_r)
        #assert len(cells_l) == len(cells_r)
        return (cells_l, cells_r)

    def getCellPattern_2x1(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]

        i_t = np.arange(0, cells_i - 1)
        j_t = np.arange(0, cells_j)
        cells_t = list(map(tuple, np.array(np.meshgrid(i_t, j_t, indexing='ij')).T.reshape(-1, 2).tolist()))

        i_b = np.arange(1, cells_i)
        j_b = np.arange(0, cells_j)
        cells_b = list(map(tuple, np.array(np.meshgrid(i_b, j_b, indexing='ij')).T.reshape(-1, 2).tolist()))

        #assert self._isCellValid(cells_t)
        #assert self._isCellValid(cells_b)
        #assert len(cells_t) == len(cells_b), str(len(cells_t)) + "!=" + str(len(cells_b))
        return (cells_t, cells_b)

    def getCellPattern_2x2(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]

        i_tl = np.arange(0, cells_i - 1)
        j_tl = np.arange(0, cells_j - 1)
        i_tr = np.arange(0, cells_i - 1)
        j_tr = np.arange(1, cells_j)
        i_bl = np.arange(1, cells_i)
        j_bl = np.arange(0, cells_j - 1)
        i_br = np.arange(1, cells_i)
        j_br = np.arange(1, cells_j)

        cells_tl = list(map(tuple, np.array(np.meshgrid(i_tl, j_tl, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_tr = list(map(tuple, np.array(np.meshgrid(i_tr, j_tr, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_bl = list(map(tuple, np.array(np.meshgrid(i_bl, j_bl, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_br = list(map(tuple, np.array(np.meshgrid(i_br, j_br, indexing='ij')).T.reshape(-1, 2).tolist()))

        #assert self._isCellValid(cells_tl)
        #assert self._isCellValid(cells_tr)
        #assert self._isCellValid(cells_bl)
        #assert self._isCellValid(cells_br)
        #assert len(cells_tl) == len(cells_bl) == len(cells_tr) == len(cells_br)
        return (cells_tl, cells_tr, cells_bl, cells_br)

    def getCellPattern_3x3(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]

        i_tl = np.arange(0, cells_i - 2)
        j_tl = np.arange(0, cells_j - 2)
        i_tr = np.arange(0, cells_i - 2)
        j_tr = np.arange(2, cells_j)
        i_bl = np.arange(2, cells_i)
        j_bl = np.arange(0, cells_j - 2)
        i_br = np.arange(2, cells_i)
        j_br = np.arange(2, cells_j)
        i_t = np.arange(0, cells_i - 2)
        j_t = np.arange(1, cells_j - 1)
        i_b = np.arange(2, cells_i)
        j_b = np.arange(1, cells_j - 1)
        i_l = np.arange(1, cells_i - 1)
        j_l = np.arange(0, cells_j - 2)
        i_r = np.arange(1, cells_i - 1)
        j_r = np.arange(2, cells_j)
        i_c = np.arange(1, cells_i - 1)
        j_c = np.arange(1, cells_j - 1)

        cells_tl = list(map(tuple, np.array(np.meshgrid(i_tl, j_tl, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_tr = list(map(tuple, np.array(np.meshgrid(i_tr, j_tr, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_bl = list(map(tuple, np.array(np.meshgrid(i_bl, j_bl, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_br = list(map(tuple, np.array(np.meshgrid(i_br, j_br, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_t = list(map(tuple, np.array(np.meshgrid(i_t, j_t, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_b = list(map(tuple, np.array(np.meshgrid(i_b, j_b, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_l = list(map(tuple, np.array(np.meshgrid(i_l, j_l, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_r = list(map(tuple, np.array(np.meshgrid(i_r, j_r, indexing='ij')).T.reshape(-1, 2).tolist()))
        cells_c = list(map(tuple, np.array(np.meshgrid(i_c, j_c, indexing='ij')).T.reshape(-1, 2).tolist()))

        #assert self._isCellValid(cells_tl)
        #assert self._isCellValid(cells_tr)
        #assert self._isCellValid(cells_bl)
        #assert self._isCellValid(cells_br)
        #assert self._isCellValid(cells_t)
        #assert self._isCellValid(cells_b)
        #assert self._isCellValid(cells_l)
        #assert self._isCellValid(cells_r)
        #assert self._isCellValid(cells_c)
        assert len(cells_tl) == len(cells_bl) == len(cells_tr) == len(cells_br) == \
               len(cells_t) == len(cells_b) == len(cells_l) == len(cells_r) == \
               len(cells_c)

        return (cells_tl, cells_t, cells_tr, cells_l, cells_c, cells_r, cells_bl, cells_b, cells_br)


##==============================================================================
class LatticeScalar(Lattice):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, shape=(0, 0), init_value=0):
        Lattice.__init__(self, dimensions=dimensions, shape=shape, dtype=float)
        self.fill(init_value)
        return

    @classmethod
    def fromPGM(cls, pgm_file, byteorder='>'):
        with open(pgm_file, 'rb') as f:
            buffer = f.read()
            try:
                header, width, height, maxval = re.search(
                    b"(^P5\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer
                ).groups()
            except AttributeError:
                raise ValueError("Not a raw PGM file: '%s'" % pgm_file)

            shape = (int(height), int(width))
            instance = cls(dimensions=2, shape=shape)
            instance.loadPGM(pgm_file, byteorder)
            return instance

    ## METHODS -----------------------------------------------------------------

    def loadPGM(self, pgm_file, byteorder='>'):
        assert self.dimensions == 2
        with open(pgm_file, 'rb') as f:
            buffer = f.read()
            try:
                header, width, height, maxval = re.search(
                    b"(^P5\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer
                ).groups()
            except AttributeError:
                raise ValueError("Not a raw PGM file: '%s'" % pgm_file)

            matrix = np.frombuffer(
                buffer,
                dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                count=int(width) * int(height),
                offset=len(header)
            ).reshape((int(height), int(width)))

            self.loadMatrix(matrix)
        return self

    """
    def toArray(self):
        array = np.asarray(self.toMatrix()).reshape(-1, order='F')
        assert array.shape[0] == self._num_cells
        return array

    """

    def invertValues(self):
        vmax = self.max()
        vmin = self.min()
        self._data = vmax - self._data + vmin
        return self

    def max(self):
        return self._data.max()

    def min(self):
        return self._data.min()

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def plot(self, vmax=0, mask=0, save="", z='', scale="lin"):
        if self.dimensions == 2:
            plotScalarField(self._data, vmax=vmax, mask=mask, save=save, scale=scale)
        elif self.dimensions == 3:
            if z == '':
                plotScalar3DField(self._data, vmax=vmax, mask=mask, save=save)
            elif 0 <= z < self.shape[2]:
                plotScalarField(self._data[:, :, z], vmax=vmax, mask=mask, save=save)
        else:
            assert False
        return self

    def _normalize(self):
        vmax = self.max()
        vmin = self.min()
        self._data = (self._data - vmin) / (vmax - vmin)
        return self

    """
    def _getMaxCell(self):
        cell = np.unravel_index(np.argmax(self.toMatrix(), axis=None), self.shape)
        assert type(cell) == tuple  # cell = (cell[0], cell[1])
        assert len(cell) == self.dimensions
        return cell

    def _getMinCell(self):
        cell = np.unravel_index(np.argmin(self.toMatrix(), axis=None), self.shape)
        assert type(cell) == tuple  # cell = (cell[0], cell[1])
        assert len(cell) == self.dimensions
        return cell
    """


##==============================================================================
class LatticeVector(Lattice):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, shape, init_value=0):
        Lattice.__init__(self, dimensions=dimensions, shape=shape, dtype=(float, dimensions))

        if init_value != 0:
            assert type(init_value) == tuple
            assert len(init_value) == self.dimensions
            self.fill(init_value)

        return

    @classmethod
    def fromMatrix(cls, matrix_i, matrix_j):
        instance = cls.fromMatrix(cls, (matrix_i, matrix_j))
        assert instance._dimensions == 2
        return instance

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _normalize(self):
        norm = np.linalg.norm(self._data)
        np.where(norm <= 1e-10, 1e-10, norm)  # Prevent division by 0
        self._data /= norm
        return self

    def _getMaxCell(self):
        assert False

    def plot(self, interpol=3, scale=15, vmax=0, mask=1, save=""):
        if self.dimensions == 2:
            plotVectorField(self._data[:, :, 0],
                            self._data[:, :, 1],
                            interpol=interpol,
                            vmax=vmax,
                            scale=scale,
                            mask=mask,
                            save=save)
        else:
            assert False
        return self

    ##==============================================================================

    """
    def __getIndicesPattern_1x1(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]
        num_cells = cells_i * cells_j
        index_c = [i for i in range(0, num_cells)]
        return np.array(index_c)

    def __getIndicesPattern_1x2(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]
        num_cells = cells_i * cells_j
        index_l = [i for i in range(0, num_cells - cells_i)]
        index_r = [i for i in range(cells_i, num_cells)]
        return np.array((index_l, index_r))

    def __getIndicesPattern_2x1(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]
        num_cells = cells_i * cells_j
        index_t = [i for i in range(0, num_cells) if (i + 1) % cells_i]
        index_b = [i for i in range(0, num_cells) if (i) % cells_i]
        return np.array((index_t, index_b))

    def __getIndicesPattern_2x2(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]
        num_cells = cells_i * cells_j
        index_tl = [i for i in range(0, num_cells - cells_i) if (i + 1) % cells_i]
        index_tr = [i for i in range(cells_i, num_cells) if (i + 1) % cells_i]
        index_bl = [i for i in range(0, num_cells - cells_i) if (i) % cells_i]
        index_br = [i for i in range(cells_i, num_cells) if (i) % cells_i]
        assert len(index_tl) == len(index_tr) == len(index_bl) == len(index_br) == (cells_i - 1) * (cells_j - 1)
        return np.array((index_tl, index_tr, index_bl, index_br))

    def __getIndicesPattern_3x3(self):
        assert self.dimensions == 2
        cells_i = self.shape[0]
        cells_j = self.shape[1]
        num_cells = cells_i * cells_j
        index_tl = [i for i in range(0, num_cells - 2 * cells_i) if (i + 1) % cells_i and (i + 2) % cells_i]
        index_t = [i for i in range(cells_i, num_cells - cells_i) if (i + 1) % cells_i and (i + 2) % cells_i]
        index_tr = [i for i in range(2 * cells_i, num_cells) if (i + 1) % cells_i and (i + 2) % cells_i]
        index_l = [i for i in range(0, num_cells - 2 * cells_i) if (i) % cells_i and (i + 1) % cells_i]
        index_c = [i for i in range(cells_i, num_cells - cells_i) if (i + 1) % cells_i and (i) % cells_i]
        index_r = [i for i in range(2 * cells_i, num_cells) if (i) % cells_i and (i + 1) % cells_i]
        index_bl = [i for i in range(0, num_cells - 2 * cells_i) if (i - 1) % cells_i and (i) % cells_i]
        index_b = [i for i in range(cells_i, num_cells - cells_i) if (i - 1) % cells_i and (i) % cells_i]
        index_br = [i for i in range(2 * cells_i, num_cells) if (i - 1) % cells_i and (i) % cells_i]
        return np.array((index_tl, index_t, index_tr, index_l, index_c, index_r, index_bl, index_b, index_br))

    def _getIndicesPattern(self, shape):
        assert self.dimensions == 2

        if shape == (1, 1) or shape == 1 or shape == (1):
            return self.__getIndicesPattern_1x1()

        elif shape == (2, 1):
            return self.__getIndicesPattern_2x1()

        elif shape == (1, 2):
            return self.__getIndicesPattern_1x2()

        elif shape == (2, 2):
            return self.__getIndicesPattern_2x2()

        elif shape == (3, 3):
            return self.__getIndicesPattern_3x3()

        else:
            assert False, "No implemented yet"
    """
