import numpy as np
from .map import DiscreteMap, DiscreteMap, DiscreteScalarMap, DiscreteScalarMap, DiscreteVectorMap
import pandas as pd
from models.gmrf.common.observation import Observation
from models.gmrf.common.lattice import Lattice, LatticeScalar
from models.gmrf.utils.report.plot import plotScalarField
import copy
import os
import scipy.sparse.linalg
import scipy.sparse


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


##============================================================================
class ObstacleMap(DiscreteScalarMap):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution=1.0, init_value=0, offset=0):
        assert 0 <= init_value <= 1
        assert type(size) == tuple
        assert len(size) == dimensions
        DiscreteScalarMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution, init_value=init_value,
                                   offset=offset)

        return

    @classmethod
    def fromPGM(cls, pgm_file, resolution=1, byteorder='>'):
        map = DiscreteScalarMap.fromPGM(pgm_file=pgm_file, resolution=resolution, byteorder=byteorder)
        map.normalize().invertValues()
        return cls.fromMatrix(map.toMatrix(), resolution)

    ## METHODS -----------------------------------------------------------------

    def loadPGM(self, pgm_file, byteorder='>'):
        assert self.dimensions == 2
        DiscreteScalarMap.loadPGM(self, pgm_file, byteorder='>')
        self.normalize().invertValues()
        return self

    ## Cell

    def _getObstacleProbabilityAtCell(self, cell):
        if type(cell) == tuple:
            probability = self.getCell(cell)
            return probability
        elif type(cell) == list:
            return [self._getObstacleProbabilityAtCell(c) for c in cell]
        else:
            assert False

    def _hasObstacleAtCell(self, cell, threshold_probability=0.5):
        if type(cell) == tuple:
            assert 0.0 <= threshold_probability <= 1.0
            return self._getObstacleProbabilityAtCell(cell) >= threshold_probability
        elif type(cell) == list:
            return [self._hasObstacleAtCell(c, threshold_probability) for c in cell]
        else:
            assert False

    def _isCellFree(self, cell, threshold=0.5):
        return not self._hasObstacleAtCell(cell, threshold)

    def _getCellFreeProbability(self, cell):
        obstacle_probability = self.getObstacleProbabilityAtCell(cell)
        free_probability = 1 - obstacle_probability
        assert 0.0 <= free_probability <= 1.0
        return free_probability

    def _getObstacleProbabilityBetweenCells(self, cell_1, cell_2):
        if type(cell_1) is tuple and type(cell_2) is tuple:
            path = self._getAllCellCoordinatesBetween(cell_1, cell_2)

            free_probability = 1
            for cell in path:
                cell = (cell[0], cell[1])
                free_probability *= (1 - self._getObstacleProbabilityAtCell(cell))

            occupied_probability = (1 - free_probability)
            assert (0.0 <= occupied_probability <= 1.0)
            return occupied_probability

        elif type(cell_1) == list and type(cell_2) == list:
            assert len(cell_1) == len(cell_2)
            obstacles = []
            for pair in zip(cell_1, cell_2):
                obstacles += [self._getObstacleProbabilityBetweenCells(pair[0], pair[1])]
            assert len(obstacles) == len(cell_1)
            return obstacles

        else:
            assert False

    def _hasObstacleBetweenCells(self, cell_1, cell_2, threshold_probability=0.5):
        assert 0.0 <= threshold_probability <= 1.0
        return self._getObstacleProbabilityBetweenCells(cell_1, cell_2) > threshold_probability

    def _areCellsConnected(self, cell_1, cell_2, threshold_probability=0.5):
        return not self._hasObstacleBetweenCells(cell_1, cell_2, threshold_probability)

    ## Position

    def getObstacleProbabilityAtPosition(self, position, fix_position=False):
        cell = self._convertPositionToCell(position, fix_position=fix_position)
        return self._getObstacleProbabilityAtCell(cell)

    def hasObstacleAtPosition(self, position, threshold_probability=0.5, fix_position=False):
        assert 0.0 <= threshold_probability <= 1.0
        cell = self._convertPositionToCell(position, fix_position=fix_position)
        return self._hasObstacleAtCell(cell, threshold_probability)

    def isPositionFree(self, position, threshold=0.5, fix_position=False):
        return not self.hasObstacleAtPosition(position, threshold, fix_position=fix_position)

    def getObstacleProbabilityBetweenPositions(self, start_position, end_position, fix_position=False):
        start_cell = self._convertPositionToCell(start_position, fix_position=fix_position)
        end_cell = self._convertPositionToCell(end_position, fix_position=fix_position)
        return self._getObstacleProbabilityBetweenCells(start_cell, end_cell)

    def hasObstacleBetweenPositions(self, position_1, position_2, threshold_probability=0.5, fix_position=False):
        assert 0.0 <= threshold_probability <= 1.0
        return self.getObstacleProbabilityBetweenPositions(position_1, position_2,
                                                           fix_position=fix_position) > threshold_probability

    def arePositionsConnected(self, position_1, position_2, threshold_probability=0.5, fix_position=False):
        return not self.hasObstacleBetweenPositions(position_1, position_2, threshold_probability,
                                                    fix_position=fix_position)

    """
    The following are commented because I dont know if they are still in use. Some seem redundant

    def getObstacleProbabilityForVector(self, start_position, vector):
        assert False
        desired_next_position = (start_position[0] + vector[0],
                                 start_position[1] + vector[1])

        obstacle_probability = self.getObstacleProbabilityBetweenPositions(start_position,
                                                                           desired_next_position)
        return self.obstacle_probability

    def hasObstacleBetweenPositions(self, start_position, target_position):
        assert False
        path = self.getFeasiblePathBetweenPositions(start_position,
                                                    target_position)

        return not np.allclose(target_position, path[-1], self.resolution / 2)

    def hasObstacleForVector(self, start_position, vector):
        assert False
        desired_next_position = np.array(start_position) + np.array(vector)
        desired_next_position = tuple(desired_next_position.tolist())
        assert len(desired_next_position) == self.dimensions
        return self.hasObstacleBetweenPositions(start_position, desired_next_position)
    """

    """
    def getConnectivityBetweenIndices(self, index_1, index_2):
        assert (type(index_1) is int and type(index_2) is int) \
               or (type(index_1) is list and type(index_2) is list and len(index_1) == len(index_2)) \
               or (type(index_1) is np.ndarray and type(index_2) is np.ndarray and len(index_1) == len(index_2))

        connectivity = self.__connectivity[index_1, index_2]
        assert len(index_1) == len(index_2) == len(connectivity)
        return connectivity


    def getConnectivityBetweenCells(self, cell_1, cell_2):
        index_1 = self._convertCellToIndex(cell_1)
        index_2 = self._convertCellToIndex(cell_2)
        return self.getConnectivityBetweenIndices(index_1, index_2)

    def getConnectivityBetweenPositions(self, position_1, position_2):
        index_1 = self._convertPositionToIndex(position_1)
        index_2 = self._convertPositionToIndex(position_2)
        return self.getConnectivityBetweenIndices(index_1, index_2)

    def _areCellsConnected(self, cell_1, cell_2, threshold=0.5):
        assert 0.0 <= threshold <= 1.0
        return self.getConnectivityBetweenCells(cell_1, cell_2) >= threshold

    def arePositionsConnected(self, position_1, position_2, threshold=0.5):
        assert 0.0 <= threshold <= 1.0
        return self.getConnectivityBetweenPositions(position_1, position_2) >= threshold

    """

    def getFeasiblePathForVector(self, start_position, vector):
        assert len(start_position) == len(vector) == self.dimensions
        desired_next_position = tuple((np.array(start_position) + np.array(vector)).tolist())
        path = self.getFeasiblePathBetweenPositions(start_position, desired_next_position)
        return path

    ## Other formats

    def toDistanceMap(self, start_position):
        start_index = self._convertPositionToIndex(start_position)

        assert self.dimensions == 2  # Only for 2D, for now
        indices = self._getIndicesPattern((2, 2))
        i_tl = indices[0]
        i_tr = indices[1]
        i_bl = indices[2]
        i_br = indices[3]

        array = 9999 * np.ones(self._num_cells)
        array[start_index] = 0.0
        om = self.toMatrix().T.flatten()
        dist = self.resolution
        ddist = self.resolution * np.sqrt(2)

        j = 2 * max(self.shape[0], self.shape[1])
        while j > 0:
            j = j - 1
            om_mask = om > 0.5
            array[i_tl] = np.minimum(np.minimum(np.minimum(array[i_tl], array[i_bl] + dist), array[i_tr] + dist),
                                     array[i_br] + ddist)
            array[om_mask] = 9999
            array[i_tr] = np.minimum(np.minimum(np.minimum(array[i_tr], array[i_br] + dist), array[i_tl] + dist),
                                     array[i_bl] + ddist)
            array[om_mask] = 9999
            array[i_bl] = np.minimum(np.minimum(np.minimum(array[i_bl], array[i_tl] + dist), array[i_br] + dist),
                                     array[i_tr] + ddist)
            array[om_mask] = 9999
            array[i_br] = np.minimum(np.minimum(np.minimum(array[i_br], array[i_tr] + dist), array[i_bl] + dist),
                                     array[i_tl] + ddist)
            array[om_mask] = 9999

        distance_matrix = array.reshape(self.shape).T
        return DistanceMap.fromMatrix(distance_matrix, resolution=self.resolution)

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def getFeasiblePathBetweenPositions(self, start_position, end_position, step_size=-1):

        ## First, get an ideal path that goes through all cells and see if it hits an obstacle
        ## - Clip desired position to fall INSIDE the environment
        ## - Get ideal path (no obstacles considered)
        ## - Test each position in path for colision
        clipped_end_position = self._clipPosition(end_position)
        ideal_path = DiscreteScalarMap.getStraightPathBetweenPositions(self, start_position=start_position,
                                                                       end_position=clipped_end_position,
                                                                       step_size=-1)
        furthest_excursion = start_position
        for position in ideal_path:
            if not self.hasObstacleAtPosition(position):
                furthest_excursion = position
            else:
                break

        ## Given the maximum excursion, compute straight path from start to there
        obstacle_free_path = DiscreteScalarMap.getStraightPathBetweenPositions(self, start_position,
                                                                               furthest_excursion,
                                                                               step_size=step_size)
        return obstacle_free_path


##==============================================================================
class DistanceMap(DiscreteScalarMap):

    def __init__(self, dimensions, size, resolution=1.0):
        assert type(size) == tuple
        assert len(size) == dimensions
        assert resolution > 0
        DiscreteScalarMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        return

    @classmethod
    def fromObstacleMap(cls, om, start_position):
        assert type(start_position) == tuple
        assert om.dimensions == len(start_position)
        return om.toDistanceMap(start_position)

    def getShortestPathToPosition(self, position):
        """
        assert type(position) == tuple
        assert len(position) == self.dimensions
        dist = self.toMatrix()
        path = [position]
        new_position = position
        assert dist.shape[0] > 3 and dist.shape[1] > 3

        i = 0

        while self.getPosition(position) >= self.resolution / 2:
            i += 1
            cell = self._convertPositionToCell(position)

            move_cells = [self._clipCell((cell[0] + 0, cell[1] + 1)),
                          self._clipCell((cell[0] + 0, cell[1] - 1)),
                          self._clipCell((cell[0] - 1, cell[1] + 0)),
                          self._clipCell((cell[0] + 1, cell[1] + 0)),
                          self._clipCell((cell[0] - 1, cell[1] + 1)),
                          self._clipCell((cell[0] + 1, cell[1] + 1)),
                          self._clipCell((cell[0] - 1, cell[1] - 1)),
                          self._clipCell((cell[0] + 1, cell[1] - 1))
                          ]

            nearby_dist = self.getCell(move_cells)
            best_cell = move_cells[nearby_dist.index(min(nearby_dist))]
            next_position = self._convertCellToPosition(best_cell)
            path += [next_position]
            position = next_position

        path.reverse()
        return path
        """
        pass  # Disabled for now, as this was copied from 2D. have to adapt for any num of dimensions
