import sys
sys.path.append("/home/nicolas/git/gdm/")

if(__name__ is "__main__"):
	from gmrf import gmrf_gas
	from common.debug_setups.debug_setup_3 import obstacle_map, observations
	from utils.benchmarking import ExecutionTimer
	from utils.report import getInstanceSize

	"""
	t = ExecutionTimer("GMRF Init")
	g = gmrf_gas.GMRF_Gas_Efficient(obstacle_map.toCellConnectivityMap())
	g.addObservation(observations)
	t.getElapsed()

	t = ExecutionTimer("Gas estimation")
	g.estimate()
	t.getElapsed()

	t = ExecutionTimer("Gas uncertainty")
	g.computeUncertainty()
	t.getElapsed()

	#g.getGasEstimate().plot()
	#g.getGasUncertainty().plot()
	g.toNormalDistributionMap().plot()
	"""



	"""
	import numpy as np
	from gdm.common import CellConnectivityMap
	low_res_conn = np.ones((25,25))
	low_res_conn[1, 2]   = low_res_conn[2, 1]   = 0
	low_res_conn[6, 7]   = low_res_conn[7, 6]   = 0
	low_res_conn[16, 17] = low_res_conn[17, 16] = 0
	low_res_conn[21, 22] = low_res_conn[22, 21] = 0

	low_res_ccm = CellConnectivityMap.fromMatrix(low_res_conn, size=(5,5), resolution=1)

	t = ExecutionTimer("GMRF Init")
	g = gmrf_gas.GMRF_Gas_Efficient(low_res_ccm)
	g.addObservation(observations)
	t.getElapsed()

	t = ExecutionTimer("Gas estimation")
	g.estimate()
	t.getElapsed()

	t = ExecutionTimer("Gas uncertainty")
	g.computeUncertainty()
	t.getElapsed()

	g.toNormalDistributionMap().plot()
	"""





	from gdm.common.debug_setups.debug_setup_1 import om_16, om_8, observations

	t = ExecutionTimer("GMRF Init")
	g = gmrf_gas.GMRF_Gas_Efficient(om_16.toCellConnectivityMap())
	g.addObservation(observations)
	t.getElapsed()

	t = ExecutionTimer("Gas estimation")
	g.estimate()
	t.getElapsed()

	t = ExecutionTimer("Gas uncertainty")
	g.computeUncertainty()
	t.getElapsed()

	#g.getGasEstimate().plot()
	#g.getGasUncertainty().plot()
	g.toNormalDistributionMap().plot()



	t = ExecutionTimer("GMRF Init")
	g = gmrf_gas.GMRF_Gas_Efficient(om_8.toCellConnectivityMap())
	g.addObservation(observations)
	t.getElapsed()

	t = ExecutionTimer("Gas estimation")
	g.estimate()
	t.getElapsed()

	t = ExecutionTimer("Gas uncertainty")
	g.computeUncertainty()
	t.getElapsed()

	#g.getGasEstimate().plot()
	#g.getGasUncertainty().plot()
	g.toNormalDistributionMap().plot()
