import numpy as np
import scipy.sparse.linalg

from models.gmrf.gmrf import GMRF, GMRF_Gas, GMRF_Wind, GMRF_Efficient, GMRF_Gas_Efficient, GMRF_Wind_Efficient
from models.gmrf.common import Observation
from models.gmrf.gmrf.gmrf_gas import DEFAULT_SIGMA_GZ, DEFAULT_SIGMA_GR, DEFAULT_SIGMA_GB, DEFAULT_GTK
from models.gmrf.gmrf.gmrf_wind import DEFAULT_SIGMA_WZ, DEFAULT_SIGMA_WR, DEFAULT_SIGMA_WC, DEFAULT_SIGMA_WO, DEFAULT_SIGMA_WB, \
    DEFAULT_WTK

DEFAULT_RES = 0.1
DEFAULT_SIGMA_GW = 0.12 / DEFAULT_RES

##============================================================================
class GMRF_Gas_Wind(GMRF_Gas, GMRF_Wind, GMRF):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self,
                 obstacle_map,
                 sigma_gz=DEFAULT_SIGMA_GZ,
                 sigma_gr=DEFAULT_SIGMA_GR,
                 sigma_wz=DEFAULT_SIGMA_WZ,
                 sigma_wr=DEFAULT_SIGMA_WR,
                 sigma_wc=DEFAULT_SIGMA_WC,
                 sigma_wo=DEFAULT_SIGMA_WO,
                 sigma_gw=DEFAULT_SIGMA_GW,
                 sigma_gb=DEFAULT_SIGMA_GB,
                 sigma_wb=DEFAULT_SIGMA_WB,
                 gtk=DEFAULT_GTK,
                 wtk=DEFAULT_WTK,
                 max_gas_concentration=1,
                 wind_to_gas_weight=1,
                 resolution=0):

        ## Init base
        GMRF.__init__(self, dimensions=2, obstacle_map=obstacle_map, resolution=resolution)

        GMRF_Gas.__init__(self,
                          obstacle_map,
                          sigma_gz=sigma_gz,
                          sigma_gr=sigma_gr,
                          sigma_gb=sigma_gb,
                          gtk=gtk,
                          max_gas_concentration=max_gas_concentration,
                          resolution=self.resolution)

        GMRF_Wind.__init__(self,
                           obstacle_map,
                           sigma_wz=sigma_wz / wind_to_gas_weight,
                           sigma_wr=sigma_wr / wind_to_gas_weight,
                           sigma_wc=sigma_wc / wind_to_gas_weight,
                           sigma_wo=sigma_wo / wind_to_gas_weight,
                           sigma_wb=sigma_wb / wind_to_gas_weight,
                           wtk=wtk,
                           resolution=self.resolution)

        self._sigma_gw = sigma_gw * self.resolution
        return

    ## METHODS -----------------------------------------------------------------

    def getPosition(self, position):
        if type(position) == tuple:
            gas = self.getGasEstimate().getPosition(position)
            wind = self.getWindEstimate().getPosition(position, format='xy')
            return Observation(position=position, gas=gas, wind=wind, data_type='gas+wind', dimensions=2)
        elif type(position) == list:
            observations = [self.getPosition(p) for p in position]
            return observations
        else:
            assert False

    ## PRIVATE -----------------------------------------------------------------

    def _getAxb_gw(self, m):
        assert 3 * self._gas._num_cells == m.shape[0]

        row = []
        col = []
        data = []
        r = []
        n = 0
        cells_i = self._gas.shape[0]
        cells_j = self._gas.shape[1]
        num_cells = self._gas._num_cells

        for i in range(1, cells_i - 1):
            for j in range(1, cells_j - 1):
                cell = (i, j)

                # Get coordinates of neighbours
                cell_u = (i - 1, j)
                cell_d = (i + 1, j)
                cell_l = (i, j - 1)
                cell_r = (i, j + 1)

                ## Positions
                position = self._gas._convertCellToPosition(cell)
                position_u = self._gas._convertCellToPosition(cell_u)
                position_d = self._gas._convertCellToPosition(cell_d)
                position_l = self._gas._convertCellToPosition(cell_l)
                position_r = self._gas._convertCellToPosition(cell_r)

                ## Cell indices
                i_g = self._gas._convertCellToIndex(cell)
                i_g_u = self._gas._convertCellToIndex(cell_u)
                i_g_d = self._gas._convertCellToIndex(cell_d)
                i_g_l = self._gas._convertCellToIndex(cell_l)
                i_g_r = self._gas._convertCellToIndex(cell_r)
                i_w_i = self._gas._convertCellToIndex(cell) + num_cells
                i_w_j = self._gas._convertCellToIndex(cell) + 2 * num_cells

                ## Get current values of cells
                g = m[i_g]
                g_u = m[i_g_u]
                g_d = m[i_g_d]
                g_l = m[i_g_l]
                g_r = m[i_g_r]
                w_i = m[i_w_i]
                w_j = m[i_w_j]

                ## Probability of cell being free
                f_u = 1 - self._obstacle_map.getObstacleProbabilityAtPosition(position_u, fix_position=True)
                # f_d = 1 - self._obstacle_map.getObstacleProbabilityAtPosition(position_d, fix_position=True)
                f_l = 1 - self._obstacle_map.getObstacleProbabilityAtPosition(position_l, fix_position=True)
                # f_r = 1 - self._obstacle_map.getObstacleProbabilityAtPosition(position_r, fix_position=True)

                ## Averaging weights
                # inv_resolution = 1/self.resolution
                a_i = f_u  # *inv_resolution
                b_i = 0  # f_d#*inv_resolution
                a_j = f_l  # *inv_resolution
                b_j = 0  # f_r#*inv_resolution

                ## Gradient
                dg_i = (a_i * (g - g_u) + b_i * (g_d - g))
                dg_j = (a_j * (g - g_l) + b_j * (g_r - g))
                # dg_i = a_i*(-g_u) + b_i*(g_d)
                # dg_j = a_j*(-g_l) + b_j*(g_r)

                ##||dg||·||w||·cosQ
                dgw = dg_i * w_i + dg_j * w_j
                p_i = a_i - b_i
                p_j = a_j - b_j
                dr_g = (p_i * w_i + p_j * w_j)
                dr_g_u = -a_i * w_i
                dr_g_d = b_i * w_i
                dr_g_l = -a_j * w_j
                dr_g_r = b_j * w_j
                dr_w_i = dg_i
                dr_w_j = dg_j

                data += [dr_g, dr_g_u, dr_g_d, dr_g_l, dr_g_r, dr_w_i, dr_w_j]
                col += [i_g, i_g_u, i_g_d, i_g_l, i_g_r, i_w_i, i_w_j]
                row += [n, n, n, n, n, n, n]
                r += [dgw]
                n += 1

        J_gw = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 3 * num_cells))
        L_gw = scipy.sparse.diags((1 / (self._sigma_gw ** 2)) * np.ones(n))
        r_gw = scipy.sparse.csr_matrix(r).T
        return J_gw, L_gw, r_gw

    def _getAxb(self, m):
        assert 3 * self._gas._num_cells == m.shape[0]

        num_cells = self._gas._num_cells
        J_g, L_g, r_g = GMRF_Gas._getAxb(self, m[0 * num_cells: 1 * num_cells])
        J_w, L_w, r_w = GMRF_Wind._getAxb(self, m[1 * num_cells: 3 * num_cells])
        J_gw, L_gw, r_gw = self._getAxb_gw(m)

        J = scipy.sparse.bmat([[scipy.sparse.bmat([[J_g, None], [None, J_w]])], [J_gw]], format='csr')
        L = scipy.sparse.bmat([[L_g, None, None],
                               [None, L_w, None],
                               [None, None, L_gw]], format='dia')
        r = scipy.sparse.bmat([[r_g], [r_w], [r_gw]], format='csc')

        assert J.shape[1] == 3 * self._gas._num_cells
        assert J.shape[0] == L.shape[0]
        assert J.shape[0] == r.shape[0]
        assert r.shape[1] == 1
        return J, L, r

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _estimate(self):

        assert len(self._observations) > 0

        num_cells = self._gas._num_cells

        # Solve iteratively in maximum 25 steps
        m = np.zeros((3 * num_cells))
        i = 15
        stop = False
        H = 0

        while i > 0 and not stop:

            # Compute new increment and update best estimate for m
            J, L, r = self._getAxb(m)
            Jt = J.T
            H = (Jt * L * J)
            g = (Jt * L * (-r))
            delta_m = scipy.sparse.linalg.spsolve(H, g)
            m += delta_m

            # Check if error is "small enough" to force a premature stop
            if np.allclose(delta_m, np.zeros((3 * num_cells)), atol=1e-02):
                # print("Force stop")
                stop = True

            ## Decrement counter
            # print("[GAS_WIND ITERATING] " +str(i) + " Approx error: " + str(np.abs(delta_m).sum()))
            i -= 1

        ## Check stop condition
        # if not stop:
        #	print("[GMRF_GAS_WIND] Warning: prediction stopped before reaching a _good_ solution because it was taking too long")

        ## Store final result
        m_g = m[0 * num_cells: 1 * num_cells]
        m_wi = m[1 * num_cells: 2 * num_cells]
        m_wj = m[2 * num_cells: 3 * num_cells]

        gas = scipy.sparse.csc_matrix(m_g).reshape(self._gas.shape, order='F').toarray()
        wind_i = scipy.sparse.csc_matrix(m_wi).reshape(self._gas.shape, order='F').toarray()
        wind_j = scipy.sparse.csc_matrix(m_wj).reshape(self._gas.shape, order='F').toarray()

        ## GAS CAN NOT BE NEGATIVE
        if gas.min() < 0:
            # print("Warning: GMRF prediction estimates negative gas. Capping... Min value: " +str(gas.min()))
            gas[gas < 0] = 0

        self._gas.loadMatrix(gas)
        self._wind.loadMatrix((wind_i, wind_j))
        self._H = H

        return self

    def _computeUncertainty(self):

        cells_i = self._gas.shape[0]
        cells_j = self._gas.shape[1]
        num_cells = self._gas._num_cells
        H_inv_diag = np.zeros(num_cells)
        solver = scipy.sparse.linalg.factorized(self._H[0:num_cells, 0:num_cells])

        for k in range(0, num_cells):
            e_k = np.zeros(num_cells)
            e_k[k] = 1
            H_inv_diag[k] = solver(e_k)[k]

        gas_cov = H_inv_diag.reshape((cells_j, cells_i)).transpose()
        #area_gas_cov = gas_cov / self.resolution**2
        self._gas_uncertainty.loadMatrix(gas_cov)
        return self

    def _getWindUncertainty(self):
        assert False, "Not computed by _computeUncertainty"


"""
##============================================================================
class GMRF_Gas_Wind_Parallel(GMRF_Gas_Wind):

    def __init__(self,
                 cell_connectivity_map,
                 sigma_gz=DEFAULT_SIGMA_GZ,
                 sigma_gr=DEFAULT_SIGMA_GR,
                 sigma_wz=DEFAULT_SIGMA_WZ,
                 sigma_wr=DEFAULT_SIGMA_WR,
                 sigma_wc=DEFAULT_SIGMA_WC,
                 sigma_wo=DEFAULT_SIGMA_WO,
                 sigma_gw=DEFAULT_SIGMA_GW,
                 sigma_gb=DEFAULT_SIGMA_GB,
                 sigma_wb=DEFAULT_SIGMA_WB,
                 gtk=DEFAULT_GTK,
                 wtk=DEFAULT_WTK,
                 max_gas_concentration=1,
                 wind_to_gas_weight=200):
        GMRF_Gas_Wind.__init__(self, cell_connectivity_map,
                               sigma_gz=sigma_gz,
                               sigma_gr=sigma_gr,
                               sigma_wz=sigma_wz,
                               sigma_wr=sigma_wr,
                               sigma_wc=sigma_wc,
                               sigma_wo=sigma_wo,
                               sigma_gw=sigma_gw,
                               sigma_gb=sigma_gb,
                               sigma_wb=sigma_wb,
                               gtk=gtk,
                               wtk=wtk,
                               max_gas_concentration=max_gas_concentration)

        return

    def _computeUncertainty(self):
        assert False, "I'm trying to disable this manually"
        diagonal = getDiagonalOfInverse(self._H, self._gas._num_cells)
        uncertainty = diagonal.reshape(self._gas.shape).T
        self._gas_uncertainty.loadMatrix(uncertainty)
        return self

    def _triggerUpdateObstacleMap(self):
        return self

"""


##============================================================================
class GMRF_Gas_Wind_Efficient(GMRF_Gas_Wind, GMRF_Gas_Efficient, GMRF_Wind_Efficient):

    def __init__(self,
                 obstacle_map,
                 sigma_gz=DEFAULT_SIGMA_GZ,
                 sigma_gr=DEFAULT_SIGMA_GR,
                 sigma_wz=DEFAULT_SIGMA_WZ,
                 sigma_wr=DEFAULT_SIGMA_WR,
                 sigma_wc=DEFAULT_SIGMA_WC,
                 sigma_wo=DEFAULT_SIGMA_WO,
                 sigma_gw=DEFAULT_SIGMA_GW,
                 sigma_gb=DEFAULT_SIGMA_GB,
                 sigma_wb=DEFAULT_SIGMA_WB,
                 gtk=DEFAULT_GTK,
                 wtk=DEFAULT_WTK,
                 max_gas_concentration=1,
                 wind_to_gas_weight=1,
                 resolution=0):
        ## Base init
        GMRF_Gas_Wind.__init__(self,
                               obstacle_map,
                               sigma_gz=sigma_gz,
                               sigma_gr=sigma_gr,
                               sigma_wz=sigma_wz,
                               sigma_wr=sigma_wr,
                               sigma_wc=sigma_wc,
                               sigma_wo=sigma_wo,
                               sigma_gw=sigma_gw,
                               sigma_gb=sigma_gb,
                               sigma_wb=sigma_wb,
                               gtk=gtk,
                               wtk=wtk,
                               max_gas_concentration=max_gas_concentration,
                               wind_to_gas_weight=wind_to_gas_weight,
                               resolution=resolution)

        GMRF_Gas_Efficient.__init__(self,
                                    obstacle_map=self._obstacle_map,
                                    sigma_gz=sigma_gz,
                                    sigma_gr=sigma_gr,
                                    sigma_gb=sigma_gb,
                                    gtk=gtk,
                                    max_gas_concentration=max_gas_concentration,
                                    resolution=self.resolution)

        GMRF_Wind_Efficient.__init__(self,
                                     obstacle_map=self._obstacle_map,
                                     sigma_wz=sigma_wz / wind_to_gas_weight,
                                     sigma_wr=sigma_wr / wind_to_gas_weight,
                                     sigma_wc=sigma_wc / wind_to_gas_weight,
                                     sigma_wo=sigma_wo / wind_to_gas_weight,
                                     wtk=wtk,
                                     resolution=self.resolution)

        self._triggerUpdateObstacleMap()
        return

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _triggerUpdateObstacleMap(self):
        GMRF_Gas_Efficient._triggerUpdateObstacleMap(self)
        GMRF_Wind_Efficient._triggerUpdateObstacleMap(self)
        #self.__precompute_JL_gw()
        return self

"""    
    def __precompute_JL_gw(self):
        self._precomputeCells_2x2(self._wind)

        num_cells = self._wind._num_cells
        index_tl = self._indices_2x2_tl
        index_tr = self._indices_2x2_tr
        index_bl = self._indices_2x2_bl
        index_br = self._indices_2x2_br
        
        index_g_c = index_tl
        index_g_r = index_tr
        index_g_b = index_bl
        index_w_i = np.array(index_tl) + num_cells
        index_w_j = np.array(index_tl) + 2*num_cells

        n = len(index_g_c)
        col = np.concatenate((index_g_c, index_g_r, index_g_b, index_w_i, index_w_j), axis=0)
        row = np.array(5 * [r for r in range(n)])
        data = np.concatenate((dr_dg_c, dr_dg_b, dr_dg_l, dr_dw_i, dr_dw_j), axis=0)
        J_gw = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 3 * num_cells))
        
        
 

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

        assert J_gw.shape[1] == 3 * self._gas._num_cells
        assert J_gw.shape[0] == L_gw.shape[0]
        self.__J_gw = J_gw
        self.__L_gw = L_gw
        return self



	def _getAxb_gw(self, m):
		num_cells = self._gas._num_cells
		indices = self._getIndicesPattern((2,2))
		index_c = indices[0]
		index_b = indices[2]
		index_l = indices[1]

		index_c_g = index_c
		index_b_g = index_b
		index_l_g = index_l
		index_w_i = index_c + num_cells
		index_w_j = index_c + 2*num_cells

		no_c = self._ccm.getCellWeights()[index_c]
		no_b = self._ccm.getConnectivityBetweenIndices(index_c, index_b)
		no_l = self._ccm.getConnectivityBetweenIndices(index_c, index_l)

		g_c = m[index_c_g]
		g_b = m[index_b_g]
		g_l = m[index_l_g]
		w_i = m[index_w_i]
		w_j = m[index_w_j]

		dg_i = g_b - g_c
		dg_j = g_l - g_c
		dgw  = no_c* (no_b * dg_i * w_i + no_l * dg_j * w_j)
		r_gw = scipy.sparse.csr_matrix(dgw).T


		## Compute Jacobian
		dr_dg_c = no_c * (-w_i*no_b - w_j*no_l)
		dr_dg_b = no_c * (w_i*no_b)
		dr_dg_l = no_c * (w_j*no_l)
		dr_dw_i = no_c * (dg_i*no_b)
		dr_dw_j = no_c * (dg_j*no_l)

		n    = len(index_c)
		col  = np.concatenate((index_c_g, index_b_g, index_l_g, index_w_i, index_w_j), axis=0)
		row  = np.array(5*[r for r in range(0,n)])
		data = np.concatenate((dr_dg_c, dr_dg_b, dr_dg_l, dr_dw_i, dr_dw_j), axis=0)
		J_gw = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 3*num_cells))


		## Compute Lambda only if needed
		if not self.__is_gw_valid:
			self.__is_gw_valid = True

			self.__L_gw = scipy.sparse.diags(
				(1 / (self._sigma_gw ** 2)) * np.ones(n))


		## Return
		assert J_gw.shape[1] == 3*self._gas._num_cells
		assert J_gw.shape[0] == self.__L_gw.shape[0]
		assert J_gw.shape[0] == r_gw.shape[0]
		assert r_gw.shape[1] == 1

		return J_gw, self.__L_gw, r_gw



	def _getAxb(self, m):
		assert 3 * self._gas._num_cells == m.shape[0]

		num_cells = self._gas._num_cells
		J_g,  L_g,  r_g  = GMRF_Gas_Efficient._getAxb(self,  m[0*num_cells : 1*num_cells])
		J_w,  L_w,  r_w  = GMRF_Wind_Efficient._getAxb(self, m[1*num_cells : 3*num_cells])
		J_gw, L_gw, r_gw = self._getAxb_gw(m)

		J = scipy.sparse.bmat([[scipy.sparse.bmat([[J_g, None], [None, J_w]])], [J_gw]], format='csr')
		L = scipy.sparse.bmat([[L_g, None, None],
		                       [None, L_w, None],
		                       [None, None, L_gw]], format='dia')
		r = scipy.sparse.bmat([[r_g], [r_w], [r_gw]], format='csc')


		assert J.shape[1] == 3*self._gas._num_cells
		assert J.shape[0] == L.shape[0]
		assert J.shape[0] == r.shape[0]
		assert r.shape[1] == 1
		return J, L, r


	""" """
	This approach is skippy - the derivate takes into account only every two cells
	def _getAxb_gw2(self, m):
		num_cells = self._gas._num_cells
		indices = self._getIndicesPattern((3,3))
		index_c = indices[4]
		index_t = indices[1]
		index_b = indices[7]
		index_l = indices[3]
		index_r = indices[5]
		index_c_wi = index_c + 1*num_cells
		index_c_wj = index_c + 2*num_cells
		index_c_g  = index_c
		index_t_g  = index_t
		index_b_g  = index_b
		index_l_g  = index_l
		index_r_g  = index_r

		no   = 1 - self._om.toMatrix().T.flatten()
		no_c = no[index_c]
		no_t = no[index_t]
		no_b = no[index_b]
		no_l = no[index_l]
		no_r = no[index_r]


		## Compute error
		g_c = m[index_c_g]
		g_t = m[index_t_g]
		g_b = m[index_b_g]
		g_l = m[index_l_g]
		g_r = m[index_r_g]
		w_i = m[index_c_wi]
		w_j = m[index_c_wj]

		dg_i = no_t * (g_c - g_t) + no_b * (g_b - g_c)
		dg_j = no_l * (g_c - g_l) + no_r * (g_r - g_c)
		dgw  = no_c * (dg_i * w_i + dg_j * w_j)

		r_gw = scipy.sparse.csr_matrix(dgw).T


		## Compute Jacobian
		dr_dg_c = no_c * ( (no_t - no_b)*w_i + (no_l-no_r)*w_j )
		dr_dg_t = no_c * ( -no_t * w_i )
		dr_dg_b = no_c * (  no_b * w_i )
		dr_dg_l = no_c * ( -no_l * w_j )
		dr_dg_r = no_c * (  no_r * w_j )
		dr_dw_i = no_c * ( dg_i )
		dr_dw_j = no_c * ( dg_j )

		n    = len(index_c)
		col  = np.concatenate((index_c_g, index_t_g, index_b_g, index_l_g, index_r_g, index_c_wi, index_c_wj), axis=0)
		row  = np.array(7*[r for r in range(0,n)])
		data = np.concatenate((dr_dg_c, dr_dg_t, dr_dg_b, dr_dg_l, dr_dg_r, dr_dw_i, dr_dw_j), axis=0)
		J_gw = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, 3*num_cells))


		## Compute Lambda only if needed
		if not self.__is_gw_valid:
			self.__is_gw_valid = True

			self.__L_gw = scipy.sparse.diags(
				(1 / (self._sigma_gw ** 2)) * np.ones(n))


		## Return
		assert J_gw.shape[1] == 3*self._gas._num_cells
		assert J_gw.shape[0] == self.__L_gw.shape[0]
		assert J_gw.shape[0] == r_gw.shape[0]
		assert r_gw.shape[1] == 1

		return J_gw, self.__L_gw, r_gw
	""" """
"""
