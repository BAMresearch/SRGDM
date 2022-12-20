from models.gmrf.common.gdm import NormalDistributionMapper, Observation
from models.gmrf.common import ObstacleMap
from abc import abstractmethod
import numpy as np




##==============================================================================
class GMRF(NormalDistributionMapper):

	## CONSTRUCTORS ------------------------------------------------------------

	def __init__(self, dimensions, obstacle_map, resolution):
		## Parameter check
		assert isinstance(obstacle_map, ObstacleMap)

		size = obstacle_map.size

		if resolution == 0:
			print("No resolution specified for GMRF, using same as obstacle map")
			resolution = obstacle_map.resolution

		## Init base
		NormalDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)

		## Member variables
		self._obstacle_map = obstacle_map

		return



	## Abstract ------------------------------------------------------------

	@abstractmethod
	def _triggerUpdateObstacleMap(self):
		assert False, "This functions must be updated"
		pass



	## METHODS -----------------------------------------------------------------

	def updateObstacleMap(self, obstacle_map):
		#assert(obstacle_map.resolution is self.resolution)
		#assert(obstacle_map.size is self.resolution)
		#assert(obstacle_map.shape is self.resolution)

		#self._om = obstacle_map
		#self._has_new_observations = True
		#self._is_uncertainty_valid = False
		#self._triggerUpdateObstacleMap()
		# return self
		assert False, "This class no longer uses ObstacleMaps, but CellConnectivityMaps - This functions must be updated"


	def getObstacleMap(self):
		return self._obstacle_map



	## BASE CLASS IMPLEMENTATION -----------------------------------------------

	def _updateObservations(self):
		return self


	def _reset(self):
		return self




##==============================================================================
class GMRF_Efficient(GMRF):

	## CONSTRUCTORS ------------------------------------------------------------

	def __init__(self, dimensions, obstacle_map, resolution):

		GMRF.__init__(self, dimensions=dimensions, obstacle_map=obstacle_map, resolution=resolution)
		return

	def _precomputeCells_1x1(self, map):
		if not hasattr(self, '_cells_precomputed_1x1'):
			cells_1x1 = map.getCellPattern_1x1()
			assert len(cells_1x1) == map._num_cells
			positions_1x1 = map._convertCellToPosition(cells_1x1)
			self._indices_1x1 = map._convertCellToIndex(cells_1x1)
			obstacles_1x1 = self._obstacle_map.getObstacleProbabilityAtPosition(positions_1x1, fix_position=True)
			self._connectivity_1x1 = 1 - np.array(obstacles_1x1)
			#self._connectivity_1x1 = np.ones(len(obstacles_1x1))# 1 - np.array(obstacles_1x1)
			self._cells_precomputed_1x1 = True
		return self

	def _precomputeCells_2x1(self, map):
		if not hasattr(self, '_cells_precomputed_2x1'):
			cells_2x1 = map.getCellPattern_2x1()
			positions_2x1_t = map._convertCellToPosition(cells_2x1[0])
			positions_2x1_b = map._convertCellToPosition(cells_2x1[1])
			self._indices_2x1_t = map._convertCellToIndex(cells_2x1[0])
			self._indices_2x1_b = map._convertCellToIndex(cells_2x1[1])
			obstacles_2x1 = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_2x1_t,
																					  positions_2x1_b,
																					  fix_position=True)
			self._connectivity_2x1 = 1 - np.array(obstacles_2x1)
			self._cells_precomputed_2x1 = True
		return self

	def _precomputeCells_1x2(self, map):
		if not hasattr(self, '_cells_precomputed_1x2'):
			cells_1x2 = map.getCellPattern_1x2()
			positions_1x2_l = map._convertCellToPosition(cells_1x2[0])
			positions_1x2_r = map._convertCellToPosition(cells_1x2[1])
			self._indices_1x2_l = map._convertCellToIndex(cells_1x2[0])
			self._indices_1x2_r = map._convertCellToIndex(cells_1x2[1])
			obstacles_1x2 = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_1x2_l,
																					  positions_1x2_r,
																					  fix_position=True)

			self._connectivity_1x2 = 1 - np.array(obstacles_1x2)
			self._cells_precomputed_1x2 = True
		return self

	def _precomputeCells_2x2(self, map):
		if not hasattr(self, '_cells_precomputed_2x2'):
			cells_2x2 = map.getCellPattern_2x2()
			positions_2x2_tl = map._convertCellToPosition(cells_2x2[0])
			positions_2x2_tr = map._convertCellToPosition(cells_2x2[1])
			positions_2x2_bl = map._convertCellToPosition(cells_2x2[2])
			positions_2x2_br = map._convertCellToPosition(cells_2x2[3])

			self._indices_2x2_tl = map._convertCellToIndex(cells_2x2[0])
			self._indices_2x2_tr = map._convertCellToIndex(cells_2x2[1])
			self._indices_2x2_bl = map._convertCellToIndex(cells_2x2[2])
			self._indices_2x2_br = map._convertCellToIndex(cells_2x2[3])

			obstacles_2x2_tl_tr = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_2x2_tl,
																					  positions_2x2_tr,
																					  fix_position=True)
			obstacles_2x2_tr_br = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_2x2_tr,
																					  positions_2x2_br,
																					  fix_position=True)
			obstacles_2x2_br_bl = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_2x2_br,
																					  positions_2x2_bl,
																					  fix_position=True)
			obstacles_2x2_bl_tl = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_2x2_bl,
																					  positions_2x2_tl,
																					  fix_position=True)
			connectivity_2x2_tl_tr = 1 - np.array(obstacles_2x2_tl_tr)
			connectivity_2x2_tr_br = 1 - np.array(obstacles_2x2_tr_br)
			connectivity_2x2_br_bl = 1 - np.array(obstacles_2x2_br_bl)
			connectivity_2x2_bl_tl = 1 - np.array(obstacles_2x2_bl_tl)
			self._connectivity_2x2 = 1 - connectivity_2x2_tl_tr * connectivity_2x2_tr_br * connectivity_2x2_br_bl * connectivity_2x2_bl_tl
			self._cells_precomputed_2x2 = True

		return self

	def _precomputeCells_3x3(self, map):
		if not hasattr(self, '_cells_precomputed_3x3'):
			cells_3x3 = map.getCellPattern_3x3()
			positions_3x3_tl = map._convertCellToPosition(cells_3x3[0])
			positions_3x3_t = map._convertCellToPosition(cells_3x3[1])
			positions_3x3_tr = map._convertCellToPosition(cells_3x3[2])
			positions_3x3_l = map._convertCellToPosition(cells_3x3[3])
			positions_3x3_c = map._convertCellToPosition(cells_3x3[4])
			positions_3x3_r = map._convertCellToPosition(cells_3x3[5])
			positions_3x3_bl = map._convertCellToPosition(cells_3x3[6])
			positions_3x3_b = map._convertCellToPosition(cells_3x3[7])
			positions_3x3_br = map._convertCellToPosition(cells_3x3[8])

			self._indices_3x3_tl = map._convertCellToIndex(cells_3x3[0])
			self._indices_3x3_t = map._convertCellToIndex(cells_3x3[1])
			self._indices_3x3_tr = map._convertCellToIndex(cells_3x3[2])
			self._indices_3x3_l = map._convertCellToIndex(cells_3x3[3])
			self._indices_3x3_c = map._convertCellToIndex(cells_3x3[4])
			self._indices_3x3_r = map._convertCellToIndex(cells_3x3[5])
			self._indices_3x3_bl = map._convertCellToIndex(cells_3x3[6])
			self._indices_3x3_b = map._convertCellToIndex(cells_3x3[7])
			self._indices_3x3_br = map._convertCellToIndex(cells_3x3[8])

			self._obstacles_3x3_c = self._obstacle_map.getObstacleProbabilityAtPosition(positions_3x3_c,
																								fix_position=True)
			self._obstacles_3x3_c_t = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_3x3_c,
																					  positions_3x3_t,
																					  fix_position=True)
			self._obstacles_3x3_c_b = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_3x3_c,
																					  positions_3x3_b,
																					  fix_position=True)
			self._obstacles_3x3_c_l = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_3x3_c,
																					  positions_3x3_l,
																					  fix_position=True)
			self._obstacles_3x3_c_r = self._obstacle_map.getObstacleProbabilityBetweenPositions(positions_3x3_c,
																					  positions_3x3_r,
																					  fix_position=True)
			#self._connectivity_3x3_c_t = 1 - np.array(obstacles_3x3_c_t)
			#self._connectivity_3x3_c_b = 1 - np.array(obstacles_3x3_c_b)
			#self._connectivity_3x3_c_l = 1 - np.array(obstacles_3x3_c_l)
			#self._connectivity_3x3_c_r = 1 - np.array(obstacles_3x3_c_r)

			self._cells_precomputed_3x3 = True

		return self