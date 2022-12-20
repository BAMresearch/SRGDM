import numpy as np
import scipy.sparse.linalg
from models.gmrf.common import DiscreteVectorMap, NormalWindDistributionMapper
from models.gmrf.gmrf.gmrf import GMRF, GMRF_Efficient

##==============================================================================

DEFAULT_RES = 0.1
DEFAULT_SIGMA_WZ = 0.1 / DEFAULT_RES
DEFAULT_SIGMA_WR = 8.253 / DEFAULT_RES
DEFAULT_SIGMA_WC = 0.049 / DEFAULT_RES
DEFAULT_SIGMA_WO = 0.22 / DEFAULT_RES
DEFAULT_SIGMA_WB = 1000
DEFAULT_WTK = 0.012  # 10 minutes approx to drop to 10%

DEFAULT_RES = 0.1
DEFAULT_SIGMA_WZ = 0.1 / DEFAULT_RES
DEFAULT_SIGMA_WR = 0.825 / DEFAULT_RES
DEFAULT_SIGMA_WC = 0.048 / DEFAULT_RES
DEFAULT_SIGMA_WO = 0.22 / DEFAULT_RES
DEFAULT_SIGMA_WB = 1000
DEFAULT_WTK = 0.012  # 10 minutes approx to drop to 10%


##==============================================================================
class GMRF_Wind(GMRF, NormalWindDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self,
                 obstacle_map,
                 sigma_wz=DEFAULT_SIGMA_WZ,
                 sigma_wr=DEFAULT_SIGMA_WR,
                 sigma_wc=DEFAULT_SIGMA_WC,
                 sigma_wo=DEFAULT_SIGMA_WO,
                 sigma_wb=DEFAULT_SIGMA_WB,
                 wtk=DEFAULT_WTK,
                 resolution=0):

        ## Parameter check
        assert (sigma_wz > 0)
        assert (sigma_wr > 0)
        assert (sigma_wc > 0)
        assert (sigma_wo > 0)
        assert (sigma_wb > 0)
        assert (wtk >= 0)

        ## Init base
        GMRF.__init__(self, dimensions=2, obstacle_map=obstacle_map, resolution=resolution)
        NormalWindDistributionMapper.__init__(self, dimensions=self.dimensions, size=self.size,
                                              resolution=self.resolution)

        ## Member variables
        self._sigma_wz = sigma_wz * self.resolution
        self._sigma_wr = sigma_wr * self.resolution
        self._sigma_wo = sigma_wo * self.resolution
        self._sigma_wc = sigma_wc * self.resolution
        self._sigma_wb = sigma_wb
        self._wtk = wtk

        map_offset = (self.resolution, self.resolution)
        map_size = tuple((np.array(self.size) + 2 * np.array(map_offset)).tolist())
        map_dimensions = self.dimensions
        map_resolution = self.resolution
        self._wind = DiscreteVectorMap(dimensions=map_dimensions, size=map_size, resolution=map_resolution,
                                       init_value=(0, 0), offset=map_offset)
        self._wind_uncertainty = DiscreteVectorMap(dimensions=map_dimensions, size=map_size, resolution=map_resolution,
                                                   init_value=(float("+inf"), float("+inf")), offset=map_offset)
        return

    ## PRIVATE -----------------------------------------------------------------

    def _getAxb_wr(self, m):
        row = []
        col = []
        data = []
        r = []
        n = 0
        cells_i = self._wind.shape[0]
        cells_j = self._wind.shape[1]
        num_variables = self._wind._num_cells

        for i in range(0, cells_i - 1):
            for j in range(0, cells_j):
                cell = (i, j)
                cell_d = (i + 1, j)
                index_i = self._wind._convertCellToIndex(cell)
                index_j = index_i + num_variables
                index_i_d = self._wind._convertCellToIndex(cell_d)
                index_j_d = index_i_d + num_variables
                position = self._wind._convertCellToPosition(cell)
                position_d = self._wind._convertCellToPosition(cell_d)

                obstacle_probability = self._obstacle_map.getObstacleProbabilityBetweenPositions(position, position_d,
                                                                                                 fix_position=True)
                conn = 1 - obstacle_probability
                assert 0 <= conn <= 1

                k = conn
                row += [n, n]
                col += [index_i, index_i_d]
                data += [k, -k]
                r += [m[index_i] - m[index_i_d]]
                n += 1
                row += [n, n]
                col += [index_j, index_j_d]
                data += [k, -k]
                r += [m[index_j] - m[index_j_d]]
                n += 1

        for i in range(0, cells_i):
            for j in range(0, cells_j - 1):
                cell = (i, j)
                cell_r = (i, j + 1)
                index_i = self._wind._convertCellToIndex(cell)
                index_j = index_i + num_variables
                index_i_r = self._wind._convertCellToIndex(cell_r)
                index_j_r = index_i_r + num_variables
                position = self._wind._convertCellToPosition(cell)
                position_r = self._wind._convertCellToPosition(cell_r)

                obstacle_probability = self._obstacle_map.getObstacleProbabilityBetweenPositions(position, position_r,
                                                                                                 fix_position=True)
                conn = 1 - obstacle_probability
                assert 0 <= conn <= 1

                k = conn
                row += [n, n]
                col += [index_i, index_i_r]
                data += [k, -k]
                r += [m[index_i] - m[index_i_r]]
                n += 1
                row += [n, n]
                col += [index_j, index_j_r]
                data += [k, -k]
                r += [m[index_j] - m[index_j_r]]
                n += 1

        J_wr = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, self._wind._num_cells * 2))
        L_wr = (1 / (self._sigma_wr ** 2)) * np.ones(n)  # np.array(k)
        r_wr = scipy.sparse.csr_matrix(r).T

        return J_wr, L_wr, r_wr

    def _getAxb_wb(self, m):
        var_wb = self._sigma_wb ** 2

        cells_i = self._wind.shape[0]
        cells_j = self._wind.shape[1]
        connectivity_weights = np.ones(self._wind._num_cells)

        for i in range(0, cells_i):
            for j in range(0, cells_j):
                cell = (i, j)
                index = self._wind._convertCellToIndex(cell)
                position = self._wind._convertCellToPosition(cell)

                obstacle = self._obstacle_map.getObstacleProbabilityAtPosition(position, fix_position=True)
                free = 1 - obstacle
                connectivity_weights[index] = free

        J_wb = scipy.sparse.identity(self._wind._num_cells * 2)
        L_wb_diagonal_i = (1 / (var_wb * (connectivity_weights ** 2) + 1e-10)) * np.ones(self._wind._num_cells)
        L_wb_diagonal_j = L_wb_diagonal_i
        L_wb = np.concatenate((L_wb_diagonal_i, L_wb_diagonal_j))
        r_wb = scipy.sparse.csr_matrix(m).T

        return J_wb, L_wb, r_wb

    def _getAxb_wc(self, m):
        row = []
        col = []
        data = []
        r = []
        n = 0
        k = []

        cells_i = self._wind.shape[0]
        cells_j = self._wind.shape[1]
        num_variables = self._wind._num_cells

        for i in range(0, cells_i - 1):
            for j in range(0, cells_j - 1):
                cell = (i, j)
                cell_d = (i + 1, j)
                cell_r = (i, j + 1)
                cell_dr = (i + 1, j + 1)

                index_i = self._wind._convertCellToIndex(cell)
                index_d_i = self._wind._convertCellToIndex(cell_d)
                index_r_i = self._wind._convertCellToIndex(cell_r)
                index_dr_i = self._wind._convertCellToIndex(cell_dr)
                index_j = index_i + num_variables
                index_d_j = index_d_i + num_variables
                index_r_j = index_r_i + num_variables
                index_dr_j = index_dr_i + num_variables

                new_data = []
                new_col = []
                new_row = []
                new_r = 0.0

                """
                position = self._wind._convertCellToPosition(cell)
                position_d = self._wind._convertCellToPosition(cell_d)
                position_r = self._wind._convertCellToPosition(cell_r)
                position_dr = self._wind._convertCellToPosition(cell_dr)
                
                ## Only if all 4 cells are connected.
                ## If there are obstacles, the weight will drop to 0
                obs_c_d = self._obstacle_map.getObstacleProbabilityBetweenPositions(position, position_d,
                                                                                    fix_position=True)
                obs_d_dr = self._obstacle_map.getObstacleProbabilityBetweenPositions(position_d, position_dr,
                                                                                     fix_position=True)
                obs_dr_r = self._obstacle_map.getObstacleProbabilityBetweenPositions(position_dr, position_r,
                                                                                     fix_position=True)
                obs_r_c = self._obstacle_map.getObstacleProbabilityBetweenPositions(position_r, position,
                                                                                    fix_position=True)
                free = (1 - obs_c_d) * (1 - obs_d_dr) * (1 - obs_dr_r) * (1 - obs_r_c)
                assert 0 <= free <= 1
                """
                free = 1

                row += [n, n, n, n, n, n, n, n]
                col += [index_i, index_j, index_d_i, index_d_j, index_r_i, index_r_j, index_dr_i, index_dr_j]
                # data += [1, 1, -1, 1, 1, -1, -1, -1]
                data += (np.array([1, 1, -1, 1, 1, -1, -1, -1]) * free).tolist()
                r += [m[index_i] + m[index_j] - m[index_d_i] + m[index_d_j] + m[index_r_i] - m[index_r_j] - m[
                    index_dr_i] - m[index_dr_j]]
                n += 1
                k += [1]

        J_wc = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 2 * num_variables))
        L_wc = (1 / (self._sigma_wc ** 2)) * np.array(k)
        r_wc = scipy.sparse.csr_matrix(r).T
        return J_wc, L_wc, r_wc

    def _getAxb_wo(self, m):
        row = []
        col = []
        data = []
        r = []
        n = 0
        cells_i = self._wind.shape[0]
        cells_j = self._wind.shape[1]
        num_variables = self._wind._num_cells

        for i in range(1, cells_i - 1):
            for j in range(1, cells_j - 1):
                cell = (i, j)
                position = self._wind._convertCellToPosition(cell)
                index_i = self._wind._convertCellToIndex(cell)
                index_j = index_i + num_variables
                has_obstacle = self._obstacle_map.hasObstacleAtPosition(position, fix_position=True)

                if has_obstacle:
                    row += [n]
                    col += [index_i]
                    data += [1]
                    r += [m[index_i]]
                    n += 1
                    row += [n]
                    col += [index_j]
                    data += [1]
                    r += [m[index_j]]
                    n += 1

                else:
                    cell_u = (i - 1, j)
                    cell_d = (i + 1, j)
                    cell_l = (i, j - 1)
                    cell_r = (i, j + 1)

                    position_u = self._wind._convertCellToPosition(cell_u)
                    position_d = self._wind._convertCellToPosition(cell_d)
                    position_l = self._wind._convertCellToPosition(cell_l)
                    position_r = self._wind._convertCellToPosition(cell_r)

                    new_r = 0.0

                    ## PROBLEM!!!!! En concreto, con CCM coarse que no tienen obstaculos. TODAS las celdas estan libres.
                    ## Se arregla facil, solo hay que ver si la celda tiene conexión hacia arriba, abajo, lados, (no hay restriccion) y en caso contrario, o si es una celda borde, SI hay restriccion = 0
                    has_obstacle_u = self._obstacle_map.hasObstacleBetweenPositions(position, position_u,
                                                                                    fix_position=True)
                    has_obstacle_d = self._obstacle_map.hasObstacleBetweenPositions(position, position_d,
                                                                                    fix_position=True)
                    has_obstacle_l = self._obstacle_map.hasObstacleBetweenPositions(position, position_l,
                                                                                    fix_position=True)
                    has_obstacle_r = self._obstacle_map.hasObstacleBetweenPositions(position, position_r,
                                                                                    fix_position=True)
                    has_vertical_obstacle = has_obstacle_u or has_obstacle_d
                    has_horizontal_obstacle = has_obstacle_l or has_obstacle_r

                    if has_vertical_obstacle:
                        row += [n]
                        col += [index_i]
                        data += [1]
                        new_r += m[index_i]

                    if has_horizontal_obstacle:
                        row += [n]
                        col += [index_j]
                        data += [1]
                        new_r += m[index_j]

                    n += 1
                    r += [new_r]

        J_wo = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 2 * num_variables))
        L_wo = (1 / (self._sigma_wo ** 2)) * np.ones(n)
        b_wo = scipy.sparse.csr_matrix(r).T
        return J_wo, L_wo, b_wo

    def _getAxb_wz(self, m):
        row = []
        col = []
        data = []
        r = []
        var = []
        n = 0
        var_wz = self._sigma_wz ** 2

        for observation in self._observations:
            if observation.hasWind():
                if self._obstacle_map.isPositionFree(observation.position):
                    index_i = self._wind._convertPositionToIndex(observation.position)
                    index_j = index_i + self._wind._num_cells
                    wind_x = observation.wind[0]
                    wind_y = observation.wind[1]
                    wind_i = -wind_y
                    wind_j = wind_x

                    row += [n]
                    col += [index_i]
                    data += [-1]
                    r += [wind_i - m[index_i]]
                    n += 1
                    var += [1 / (var_wz + observation.time * self._wtk)]

                    row += [n]
                    col += [index_j]
                    data += [-1]
                    r += [wind_j - m[index_j]]
                    n += 1
                    var += [1 / (var_wz + observation.time * self._wtk)]

        J_wz = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, self._wind._num_cells * 2))
        L_wz = np.array(var)
        r_wz = scipy.sparse.csr_matrix(r).T

        return J_wz, L_wz, r_wz

    def _getAxb(self, m):
        assert 2 * self._wind._num_cells == m.shape[0]

        J_wr, L_wr, r_wr = self._getAxb_wr(m)
        J_wb, L_wb, r_wb = self._getAxb_wb(m)
        J_wc, L_wc, r_wc = self._getAxb_wc(m)
        J_wo, L_wo, r_wo = self._getAxb_wo(m)
        J_wz, L_wz, r_wz = self._getAxb_wz(m)

        J = scipy.sparse.bmat([[J_wr], [J_wb], [J_wc], [J_wo], [J_wz]], format='csr')
        L = scipy.sparse.diags(np.concatenate((L_wr, L_wb, L_wc, L_wo, L_wz)))
        r = scipy.sparse.bmat([[r_wr], [r_wb], [r_wc], [r_wo], [r_wz]], format='csc')

        assert J.shape[1] == 2 * self._wind._num_cells
        assert J.shape[0] == L.shape[0]
        assert J.shape[0] == L.shape[1]
        assert J.shape[0] == r.shape[0]
        assert r.shape[1] == 1

        return J, L, r

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _estimate(self):
        num_cells = self._wind._num_cells
        m = np.zeros(self._wind._num_cells * 2)
        J, L, r = self._getAxb(m)

        Jt = J.T
        H = (Jt * L * J)
        g = (Jt * L * (-r))
        delta_m = scipy.sparse.linalg.spsolve(H, g)
        m += delta_m

        cells_i = self._wind.shape[0]
        cells_j = self._wind.shape[1]

        m_i = m[0 * num_cells: 1 * num_cells]
        m_j = m[1 * num_cells: 2 * num_cells]
        wind_i = scipy.sparse.csc_matrix(m_i).reshape((cells_i, cells_j), order='F').toarray()
        wind_j = scipy.sparse.csc_matrix(m_j).reshape((cells_i, cells_j), order='F').toarray()
        self._wind.loadMatrix((wind_i, wind_j))
        self._H = H

        return self

    def _getWindEstimate(self):
        estimate = self._wind[1:self._wind.shape[0] - 1, 1:self._wind.shape[1] - 1]

        ## Ugly hack to avoid aliasing
        assert self.size[0] - self.resolution <= estimate.size[0] <= self.size[0] + self.resolution
        assert self.size[1] - self.resolution <= estimate.size[1] <= self.size[1] + self.resolution

        estimate.size = self.size

        return estimate

    def _getWindUncertainty(self):
        uncertainty = self._wind_uncertainty[1:self._wind.shape[0] - 1, 1:self._wind.shape[1] - 1]

        assert self.size[0] - self.resolution / 2 <= uncertainty.size[0] <= self.size[0] + self.resolution / 2
        assert self.size[1] - self.resolution / 2 <= uncertainty.size[1] <= self.size[1] + self.resolution / 2
        uncertainty.size = self.size

        return uncertainty

    def _computeUncertainty(self):
        num_variables = 2 * self._wind._num_cells
        solver = scipy.sparse.linalg.factorized(self._H)
        diagonal = np.zeros(num_variables)

        for i in range(0, num_variables):
            e_c = np.zeros(num_variables)
            e_c[i] = 1
            diagonal[i] = solver(e_c)[i]

        uncertainty_i = diagonal[0 * self._wind._num_cells:1 * self._wind._num_cells].reshape(self._wind.shape[1],
                                                                                              self._wind.shape[0]).T
        uncertainty_j = diagonal[1 * self._wind._num_cells:2 * self._wind._num_cells].reshape(self._wind.shape[1],
                                                                                              self._wind.shape[0]).T
        self._wind_uncertainty.loadMatrix(uncertainty_i, uncertainty_j)
        return self

    def _triggerUpdateObstacleMap(self):
        return self


##==============================================================================
class GMRF_Wind_Efficient(GMRF_Wind, GMRF_Efficient):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self,
                 obstacle_map,
                 sigma_wz=DEFAULT_SIGMA_WZ,
                 sigma_wr=DEFAULT_SIGMA_WR,
                 sigma_wc=DEFAULT_SIGMA_WC,
                 sigma_wo=DEFAULT_SIGMA_WO,
                 sigma_wb=DEFAULT_SIGMA_WB,
                 wtk=DEFAULT_WTK,
                 resolution=0):
        ## Init base
        GMRF_Wind.__init__(self,
                           obstacle_map,
                           sigma_wz=sigma_wz,
                           sigma_wr=sigma_wr,
                           sigma_wc=sigma_wc,
                           sigma_wo=sigma_wo,
                           sigma_wb=sigma_wb,
                           wtk=wtk,
                           resolution=resolution)
        GMRF_Efficient.__init__(self,
                                dimensions=self.dimensions,
                                obstacle_map=self._obstacle_map,
                                resolution=self.resolution)

        self._triggerUpdateObstacleMap()

        return

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _triggerUpdateObstacleMap(self):
        self.__precompute_JL_wr()
        self.__precompute_JL_wb()
        self.__precompute_JL_wc()
        self.__precompute_JL_wo()
        return self

    def _getAxb_wr(self, m):
        num_cells = self._wind._num_cells
        assert m.shape[0] == 2 * num_cells
        J_wr = self.__J_wr
        L_wr = self.__L_wr

        index_i_t = self._indices_2x1_t
        index_i_b = self._indices_2x1_b
        index_i_l = self._indices_1x2_l
        index_i_r = self._indices_1x2_r
        index_j_t = np.array(self._indices_2x1_t) + num_cells
        index_j_b = np.array(self._indices_2x1_b) + num_cells
        index_j_l = np.array(self._indices_1x2_l) + num_cells
        index_j_r = np.array(self._indices_1x2_r) + num_cells

        r_wr = scipy.sparse.csr_matrix(
            np.concatenate(((m[index_i_t] - m[index_i_b]),
                            (m[index_i_l] - m[index_i_r]),
                            (m[index_j_t] - m[index_j_b]),
                            (m[index_j_l] - m[index_j_r])),
                           axis=0)).T

        ## Return
        assert J_wr.shape[1] == 2 * self._wind._num_cells
        assert J_wr.shape[0] == L_wr.shape[0]
        assert J_wr.shape[0] == r_wr.shape[0]
        assert r_wr.shape[1] == 1
        return J_wr, L_wr, r_wr

    def _getAxb_wb(self, m):
        assert 2 * self._wind._num_cells == m.shape[0]
        J_wb = self.__J_wb
        L_wb = self.__L_wb
        r_wb = scipy.sparse.csr_matrix(m).T

        ## Return
        assert J_wb.shape[1] == 2 * self._wind._num_cells
        assert J_wb.shape[0] == L_wb.shape[0]
        assert J_wb.shape[0] == r_wb.shape[0]
        assert r_wb.shape[1] == 1
        return J_wb, L_wb, r_wb

    def _getAxb_wc(self, m):
        num_cells = self._wind._num_cells
        index_tl_i = self._indices_2x2_tl
        index_tr_i = self._indices_2x2_tr
        index_bl_i = self._indices_2x2_bl
        index_br_i = self._indices_2x2_br
        index_tl_j = np.array(self._indices_2x2_tl) + num_cells
        index_tr_j = np.array(self._indices_2x2_tr) + num_cells
        index_bl_j = np.array(self._indices_2x2_bl) + num_cells
        index_br_j = np.array(self._indices_2x2_br) + num_cells

        J_wc = self.__J_wc
        L_wc = self.__L_wc
        r_wc = scipy.sparse.csr_matrix(+ m[index_tl_i]
                                       + m[index_tr_i]
                                       - m[index_bl_i]
                                       - m[index_br_i]
                                       + m[index_tl_j]
                                       - m[index_tr_j]
                                       + m[index_bl_j]
                                       - m[index_br_j]
                                       ).T

        ## Return
        assert J_wc.shape[1] == 2 * self._wind._num_cells
        assert J_wc.shape[0] == L_wc.shape[0]
        assert J_wc.shape[0] == r_wc.shape[0]
        assert r_wc.shape[1] == 1
        return J_wc, L_wc, r_wc

    def _getAxb_wo(self, m):
        num_cells = self._wind._num_cells
        index_c_i = self._indices_3x3_c
        index_c_j = np.array(self._indices_3x3_c) + num_cells

        J_wo = self.__J_wo
        L_wo = self.__L_wo
        r_wo = scipy.sparse.csr_matrix(np.concatenate((m[index_c_i],
                                                       m[index_c_j],
                                                       m[index_c_i],
                                                       m[index_c_j]), axis=0)).T

        ## Return
        assert J_wo.shape[1] == 2 * self._wind._num_cells
        assert J_wo.shape[0] == L_wo.shape[0]
        assert J_wo.shape[0] == r_wo.shape[0]
        assert r_wo.shape[1] == 1
        return J_wo, L_wo, r_wo

    ## PRIVATE METHODS ---------------------------------------------------------
    def __precompute_JL_wr(self):
        self._precomputeCells_2x1(self._wind)
        self._precomputeCells_1x2(self._wind)
        num_cells = self._wind._num_cells
        n_tb = len(self._indices_2x1_t)  # Num pairs top-botton
        n_lr = len(self._indices_1x2_l)  # Num pairs left-right
        n_i = (n_tb + n_lr)
        n = 2 * n_i  # For Wind i and wind j, twice

        ## Vertically adjacent cells: connect if no obstacle
        col_tb = np.concatenate((self._indices_2x1_t, self._indices_2x1_b), axis=0)
        row_tb = np.array(2 * [r for r in range(0, n_tb)])
        data_tb = np.concatenate((self._connectivity_2x1, -self._connectivity_2x1), axis=0)

        ## Horizontally adjacent cells: connect if no obstacle
        col_lr = np.concatenate((self._indices_1x2_l, self._indices_1x2_r), axis=0)
        row_lr = np.array(2 * [r + n_tb for r in range(0, n_lr)])
        data_lr = np.concatenate((self._connectivity_1x2, -self._connectivity_1x2), axis=0)

        col_i = np.concatenate((col_tb, col_lr), axis=0)
        row_i = np.concatenate((row_tb, row_lr), axis=0)
        data_i = np.concatenate((data_tb, data_lr), axis=0)

        J_i = scipy.sparse.csr_matrix((data_i, (row_i, col_i)), shape=(n_i, num_cells))
        J_wr = scipy.sparse.bmat([[J_i, None], [None, J_i]], format='csr')
        L_wr = (1 / (self._sigma_wr ** 2)) * np.ones(n)

        ## Return
        assert J_wr.shape[1] == 2 * self._wind._num_cells
        assert J_wr.shape[0] == L_wr.shape[0]
        self.__J_wr = J_wr
        self.__L_wr = L_wr
        return self

    def __precompute_JL_wb(self):
        self._precomputeCells_1x1(self._wind)
        num_cells = self._wind._num_cells
        no_obs_zero = 0.5 / np.pi  # Computed for a NLML value of 0

        J_wb = scipy.sparse.identity(2 * num_cells)
        L_wb = 1 / ((self._sigma_wb ** 2) * np.concatenate((self._connectivity_1x1, self._connectivity_1x1),
                                                           axis=0) + no_obs_zero)

        assert J_wb.shape[1] == 2 * self._wind._num_cells
        assert J_wb.shape[0] == L_wb.shape[0]
        self.__J_wb = J_wb
        self.__L_wb = L_wb
        return self

    def __precompute_JL_wc(self):
        self._precomputeCells_2x2(self._wind)
        num_cells = self._wind._num_cells
        index_tl_i = self._indices_2x2_tl
        index_tr_i = self._indices_2x2_tr
        index_bl_i = self._indices_2x2_bl
        index_br_i = self._indices_2x2_br
        index_tl_j = np.array(self._indices_2x2_tl) + num_cells
        index_tr_j = np.array(self._indices_2x2_tr) + num_cells
        index_bl_j = np.array(self._indices_2x2_bl) + num_cells
        index_br_j = np.array(self._indices_2x2_br) + num_cells

        n = len(index_tl_i)
        col = np.concatenate(
            (index_tl_i, index_tr_i, index_bl_i, index_br_i, index_tl_j, index_tr_j, index_bl_j, index_br_j), axis=0)
        row = np.array(8 * [r for r in range(0, n)])
        # data = np.concatenate((+no_obs[index_tl], +no_obs[index_tr], -no_obs[index_bl],-no_obs[index_br], +no_obs[index_tl], -no_obs[index_tr],	+no_obs[index_bl], -no_obs[index_br]), axis=0)
        no = np.ones(n)  # np.array(self._connectivity_2x2)
        data = np.concatenate([+no, +no, -no, -no, +no, -no, +no, -no], axis=0)

        J_wc = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 2 * num_cells))
        L_wc = (np.sqrt(2) / (self._sigma_wc ** 2)) * np.ones(n)

        assert J_wc.shape[1] == 2 * self._wind._num_cells
        assert J_wc.shape[0] == L_wc.shape[0]
        self.__J_wc = J_wc
        self.__L_wc = L_wc
        return self

    def __precompute_JL_wo(self):
        self._precomputeCells_3x3(self._wind)

        num_cells = self._wind._num_cells
        index_c = self._indices_3x3_c
        index_t = self._indices_3x3_t
        index_b = self._indices_3x3_b
        index_l = self._indices_3x3_l
        index_r = self._indices_3x3_r
        index_c_i = index_c
        index_c_j = np.array(index_c) + num_cells

        obstacle_c = np.array(self._obstacles_3x3_c)
        obstacle_tb = 1 - (1 - np.array(self._obstacles_3x3_c_t)) * (1 - np.array(self._obstacles_3x3_c_b))
        obstacle_lr = 1 - (1 - np.array(self._obstacles_3x3_c_l)) * (1 - np.array(self._obstacles_3x3_c_r))
        n_c = 2 * len(index_c)
        n_tb = len(index_c)
        n_lr = len(index_c)

        ## Center cell: if obstacle -> make i and j 0
        col_c = np.concatenate((index_c_i, index_c_j), axis=0)
        row_c = np.arange(n_c)
        data_c = np.concatenate((obstacle_c, obstacle_c), axis=0)

        ## Vertical obstacles: if has obstacle -> make wind_i 0
        col_tb = index_c_i
        row_tb = np.array([r + n_c for r in range(n_tb)])
        data_tb = obstacle_tb

        ## Horizontal obstacles: if has obstacle -> make wind_j 0
        col_lr = index_c_j
        row_lr = np.array([r + n_c + n_tb for r in range(n_lr)])
        data_lr = obstacle_lr

        col = np.concatenate((col_c, col_tb, col_lr), axis=0)
        row = np.concatenate((row_c, row_tb, row_lr), axis=0)
        data = np.concatenate((data_c, data_tb, data_lr), axis=0)

        n = n_c + n_tb + n_lr
        J_wo = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 2 * num_cells))
        L_wo = (1 / (self._sigma_wo ** 2)) * np.ones(n)

        assert J_wo.shape[1] == 2 * self._wind._num_cells
        assert J_wo.shape[0] == L_wo.shape[0]
        self.__J_wo = J_wo
        self.__L_wo = L_wo
        return self

