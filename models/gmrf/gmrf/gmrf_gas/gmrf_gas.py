import numpy as np
import scipy.sparse.linalg
from models.gmrf.common import DiscreteScalarMap, NormalGasDistributionMapper, Observation
from models.gmrf.gmrf import GMRF, GMRF_Efficient

##============================================================================


# WARNING!!! HE TENIDO QUE ESCALAR POR LA RESOLUCION PARA EVITAR PROBLEMAS
DEFAULT_RES = 0.1
k = 0.08 # Correct for variance scale
DEFAULT_SIGMA_GZ = 0.1 / DEFAULT_RES
DEFAULT_SIGMA_GR = 1.128 / DEFAULT_RES # Compensated for resolution
DEFAULT_SIGMA_GB = k * 1000
DEFAULT_GTK = 0.012  # 10 minutes approx to drop to 10%


##============================================================================
class GMRF_Gas(GMRF, NormalGasDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self,
                 obstacle_map,
                 sigma_gz=DEFAULT_SIGMA_GZ,
                 sigma_gr=DEFAULT_SIGMA_GR,
                 sigma_gb=DEFAULT_SIGMA_GB,
                 gtk=DEFAULT_GTK,
                 max_gas_concentration=1,
                 resolution=0):

        ## Parameter check
        assert (sigma_gz > 0)
        assert (sigma_gr > 0)
        assert (sigma_gb > 0)
        assert (gtk >= 0)
        assert (max_gas_concentration > 0)

        ## Init base
        GMRF.__init__(self, dimensions=2, obstacle_map=obstacle_map, resolution=resolution)
        NormalGasDistributionMapper.__init__(self, dimensions=2, size=self.size, resolution=resolution)

        ## Member variables
        self._sigma_gz = sigma_gz * max_gas_concentration * self.resolution
        self._sigma_gr = sigma_gr * max_gas_concentration * self.resolution
        self._sigma_gb = sigma_gb * max_gas_concentration
        self._gtk = gtk
        self._H = 0

        map_offset = (self.resolution, self.resolution)
        map_size = tuple((np.array(self.size) + 2 * np.array(map_offset)).tolist())
        map_dimensions = self.dimensions
        map_resolution = self.resolution
        self._gas = DiscreteScalarMap(dimensions=map_dimensions, size=map_size, resolution=map_resolution, init_value=0,
                                      offset=map_offset)
        self._gas_uncertainty = DiscreteScalarMap(dimensions=map_dimensions, size=map_size, resolution=map_resolution,
                                                  init_value=float("+inf"), offset=map_offset)

        return

    ## PRIVATE -----------------------------------------------------------------

    def _getAxb_gr(self, m):
        assert self._gas._num_cells == m.shape[0]

        row = []
        col = []
        data = []
        r = []
        k = []
        n = 0

        cells_i = self._gas.shape[0]
        cells_j = self._gas.shape[1]
        num_variables = self._gas._num_cells

        ## Top - Bottom
        for i in range(0, cells_i - 1):
            for j in range(0, cells_j):
                cell_t = (i, j)
                cell_b = (i + 1, j)
                index_t = self._gas._convertCellToIndex(cell_t)
                index_b = self._gas._convertCellToIndex(cell_b)
                position_t = self._gas._convertCellToPosition(cell_t)
                position_b = self._gas._convertCellToPosition(cell_b)

                obstacle_probability_tb = self._obstacle_map.getObstacleProbabilityBetweenPositions(position_t, position_b,fix_position=True)
                conn_tb = 1 - obstacle_probability_tb
                assert 0 <= conn_tb <= 1

                r += [m[index_t] - m[index_b]]
                row += [n, n]
                col += [index_t, index_b]
                data += [1, -1]
                k += [conn_tb]
                n += 1

        ## Left - Right
        for i in range(0, cells_i):
            for j in range(0, cells_j - 1):
                cell_l = (i, j)
                cell_r = (i, j + 1)
                index_l = self._gas._convertCellToIndex(cell_l)
                index_r = self._gas._convertCellToIndex(cell_r)
                position_l = self._gas._convertCellToPosition(cell_l)
                position_r = self._gas._convertCellToPosition(cell_r)

                obstacle_probability_lr = self._obstacle_map.getObstacleProbabilityBetweenPositions(position_l, position_r, fix_position=True)
                conn_lr = 1 - obstacle_probability_lr
                assert 0 <= conn_lr <= 1

                r += [m[index_l] - m[index_r]]
                row += [n, n]
                col += [index_l, index_r]
                data += [1, -1]
                k += [conn_lr]
                n += 1

        J_gr = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, num_variables))
        L_gr = np.array(k) * 1 / (self._sigma_gr ** 2)
        r_gr = scipy.sparse.csr_matrix(r).T

        return J_gr, L_gr, r_gr

    def _getAxb_gb(self, m):
        assert self._gas._num_cells == m.shape[0]

        cells_i = self._gas.shape[0]
        cells_j = self._gas.shape[1]
        connectivity_weights = np.zeros(self._gas._num_cells)
        
        for i in range(0, cells_i):
            for j in range(0, cells_j):
                cell = (i, j)
                index = self._gas._convertCellToIndex(cell)
                position = self._gas._convertCellToPosition(cell)

                obstacle = self._obstacle_map.getObstacleProbabilityAtPosition(position, fix_position=True)
                free = 1 - obstacle
                connectivity_weights[index] = free

        J_gb = scipy.sparse.identity(self._gas._num_cells)
        L_gb = (1 / ((self._sigma_gb ** 2) * (connectivity_weights ** 2) + 1e-8)) * np.ones(self._gas._num_cells)
        r_gb = scipy.sparse.csr_matrix(m).T
        return J_gb, L_gb, r_gb

    def _getAxb_gz(self, m):
        assert self._gas._num_cells == m.shape[0]

        row = []
        col = []
        data = []
        r = []
        var = []
        n = 0

        for observation in self._observations:
            if observation.hasGas():
                if self._obstacle_map.isPositionFree(observation.position):
                    index = self._gas._convertPositionToIndex(observation.position)
                    r += [observation.gas - m[index]]
                    row += [n]
                    col += [index]
                    data += [-1]
                    var += [1 / (self._sigma_gz ** 2 + observation.time * self._gtk)]
                    n += 1

        J_gz = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, self._gas._num_cells))
        L_gz = np.array(var)
        r_gz = scipy.sparse.csr_matrix(r).T
        return J_gz, L_gz, r_gz

    def _getAxb(self, m):
        assert self._gas._num_cells == m.shape[0]

        J_gr, L_gr, r_gr = self._getAxb_gr(m)
        J_gb, L_gb, r_gb = self._getAxb_gb(m)
        J_gz, L_gz, r_gz = self._getAxb_gz(m)

        J = scipy.sparse.bmat([[J_gr], [J_gb], [J_gz]], format='csr')
        L = scipy.sparse.diags(np.concatenate((L_gr, L_gb, L_gz)), format='dia')
        r = scipy.sparse.bmat([[r_gr], [r_gb], [r_gz]], format='csc')

        ## Output parameter check
        assert J.shape[1] == self._gas._num_cells
        assert J.shape[0] == L.shape[0]
        assert J.shape[0] == r.shape[0]
        assert r.shape[1] == 1

        return J, L, r

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _estimate(self):

        m = np.zeros(
            self._gas._num_cells)  # Choose starting conditions. Not really relevant as it has closed solution. But useful to derive to GMRF_FAS_WIND		J, L, r = self._getAxb(m)
        J, L, r = self._getAxb(m)

        # t = ExecutionTimer("[GMRF Gas] solve")
        Jt = J.T
        H = (Jt * L * J)
        g = (Jt * L * (-r))
        delta_m = scipy.sparse.linalg.spsolve(H, g)
        m += delta_m
        assert type(m) == np.ndarray
        # t.getElapsed()

        gas_matrix = m.reshape(self._gas.shape, order='F')
        self._gas.loadMatrix(gas_matrix)
        self._H = H

        return self

    def _getGasEstimate(self):
        estimate = self._gas[1:self._gas.shape[0] - 1, 1:self._gas.shape[1] - 1]

        ## Ugly hack to avoid aliasing
        assert self.size[0] - self.resolution <= estimate.size[0] <= self.size[0] + self.resolution
        assert self.size[1] - self.resolution <= estimate.size[1] <= self.size[1] + self.resolution
        estimate.size = self.size

        return estimate

    def _getGasUncertainty(self):
        uncertainty = self._gas_uncertainty[1:self._gas.shape[0] - 1, 1:self._gas.shape[1] - 1]

        assert self.size[0] - self.resolution <= uncertainty.size[0] <= self.size[0] + self.resolution
        assert self.size[1] - self.resolution <= uncertainty.size[1] <= self.size[1] + self.resolution
        uncertainty.size = self.size

        return uncertainty

    def _computeUncertainty(self):

        num_variables = self._gas._num_cells
        solver = scipy.sparse.linalg.factorized(self._H)
        diagonal = np.zeros(num_variables)
        cells_i = self._gas.shape[0]
        cells_j = self._gas.shape[1]

        for i in range(0, num_variables):
            e_c = np.zeros(num_variables)
            e_c[i] = 1
            diagonal[i] = solver(e_c)[i]

        uncertainty = diagonal.reshape((cells_j, cells_i)).T
        self._gas_uncertainty.loadMatrix(uncertainty)
        return self

    def _getCell(self, cell):
        gas = self.getGasEstimate().getCell(cell)
        position = self._gas._convertCellToPosition(cell)
        o = Observation(position=position, gas=max(0.0, gas), data_type='gas')
        return o

    # return self

    def _triggerUpdateObstacleMap(self):
        return self


"""
##==============================================================================
class GMRF_Gas_Parallel(GMRF_Gas):

    ## ------------------------------------------------------------------------
    def __init__(self,
                 cell_connectivity_map,
                 sigma_gz=DEFAULT_SIGMA_GZ,
                 sigma_gr=DEFAULT_SIGMA_GR,
                 sigma_gb=DEFAULT_SIGMA_GB,
                 gtk=DEFAULT_GTK,
                 max_gas_concentration=1):
        GMRF_Gas.__init__(self, cell_connectivity_map, sigma_gz, sigma_gr, sigma_gb, gtk, max_gas_concentration)
        return

    def _computeUncertainty(self):
        diagonal = getDiagonalOfInverse(self._H)
        uncertainty = diagonal.reshape((self._gas.shape[1], self._gas.shape[0])).T
        self._gas_uncertainty.loadMatrix(uncertainty)
        return self
"""


##==============================================================================
class GMRF_Gas_Efficient(GMRF_Gas, GMRF_Efficient):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self,
                 obstacle_map,
                 sigma_gz=DEFAULT_SIGMA_GZ,
                 sigma_gr=DEFAULT_SIGMA_GR,
                 sigma_gb=DEFAULT_SIGMA_GB,
                 gtk=DEFAULT_GTK,
                 max_gas_concentration=1,
                 resolution=0):

        GMRF_Gas.__init__(self,
                          obstacle_map,
                          sigma_gz=DEFAULT_SIGMA_GZ,
                          sigma_gr=DEFAULT_SIGMA_GR,
                          sigma_gb=DEFAULT_SIGMA_GB,
                          gtk=DEFAULT_GTK,
                          max_gas_concentration=max_gas_concentration,
                          resolution=resolution)
        GMRF_Efficient.__init__(self,
                                dimensions=self.dimensions,
                                obstacle_map=self._obstacle_map,
                                resolution=self.resolution)

        self._triggerUpdateObstacleMap()
        return

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _triggerUpdateObstacleMap(self):
        self.__precompute_JL_gr()
        self.__precompute_JL_gb()
        return self

    def _getAxb_gr(self, m):
        assert self._gas._num_cells == m.shape[0]
        J_gr = self.__J_gr
        L_gr = self.__L_gr
        r_gr = scipy.sparse.csr_matrix(np.concatenate(((m[self._indices_2x1_t] - m[self._indices_2x1_b]),
                                                       (m[self._indices_1x2_l] - m[self._indices_1x2_r])),
                                                      axis=0)).T

        ## Return
        assert J_gr.shape[1] == self._gas._num_cells
        assert J_gr.shape[0] == L_gr.shape[0]
        assert J_gr.shape[0] == r_gr.shape[0]
        assert r_gr.shape[1] == 1
        return J_gr, L_gr, r_gr

    def _getAxb_gb(self, m):
        assert self._gas._num_cells == m.shape[0]
        J_gb = self.__J_gb
        L_gb = self.__L_gb
        r_gb = scipy.sparse.csr_matrix(m).T

        ## Return
        assert J_gb.shape[1] == self._gas._num_cells
        assert J_gb.shape[0] == L_gb.shape[0]
        assert J_gb.shape[0] == r_gb.shape[0]
        assert r_gb.shape[1] == 1
        return J_gb, L_gb, r_gb

    ## PRIVATE METHODS ---------------------------------------------------------

    def __precompute_JL_gr(self):
        self._precomputeCells_2x1(self._gas)
        self._precomputeCells_1x2(self._gas)
        num_cells = self._gas._num_cells
        n_tb = len(self._indices_2x1_t) # Num pairs top-botton
        n_lr = len(self._indices_1x2_l) # Num pairs left-right
        n = n_tb + n_lr

        ## Vertically adjacent cells: connect if no obstacle
        col_tb = np.concatenate((self._indices_2x1_t, self._indices_2x1_b), axis=0)
        row_tb = np.array(2 * [r for r in range(0, n_tb)])
        data_tb = np.concatenate((self._connectivity_2x1, -self._connectivity_2x1), axis=0)

        ## Horizontally adjacent cells: connect if no obstacle
        col_lr = np.concatenate((self._indices_1x2_l, self._indices_1x2_r), axis=0)
        row_lr = np.array(2 * [r + n_tb for r in range(0, n_lr)])
        data_lr = np.concatenate((self._connectivity_1x2, -self._connectivity_1x2), axis=0)

        col = np.concatenate((col_tb, col_lr), axis=0)
        row = np.concatenate((row_tb, row_lr), axis=0)
        data = np.concatenate((data_tb, data_lr), axis=0)

        J_gr = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, num_cells))
        L_gr = (1 / (self._sigma_gr ** 2)) * np.ones(n)

        ## Return
        assert J_gr.shape[1] == self._gas._num_cells
        assert J_gr.shape[0] == L_gr.shape[0]
        self.__J_gr = J_gr
        self.__L_gr = L_gr
        return self

    def __precompute_JL_gb(self):
        self._precomputeCells_1x1(self._gas)
        num_cells = self._gas._num_cells
        no_obs_zero = 0.5 / np.pi  # Computed for a NLML value of 0

        ## WARNGING! I create a diagonal because I assume self._index_1x1 is ordered
        #prev = -1
        #for i in self._indices_1x1:
        #    assert prev < i
        #    prev = i

        J_gb = scipy.sparse.identity(num_cells)
        L_gb = 1 / ((self._sigma_gb ** 2) * self._connectivity_1x1 + no_obs_zero)

        assert J_gb.shape[1] == self._gas._num_cells
        assert J_gb.shape[0] == L_gb.shape[0]
        self.__J_gb = J_gb
        self.__L_gb = L_gb
        return self



