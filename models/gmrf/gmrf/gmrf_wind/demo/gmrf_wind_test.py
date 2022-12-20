from gdm.gmrf import gmrf_wind
from gdm.common.debug_setups.debug_setup_2 import obstacle_map, observations
from gdm.utils.benchmarking import ExecutionTimer


if(__name__ is "__main__"):

	tt = ExecutionTimer("[GMRF-W] Total")
	t = ExecutionTimer("[GMRF-W] Initialization")
	w = gmrf_wind.GMRF_Wind_Efficient(obstacle_map.toCellConnectivityMap())
	w.addObservation(observations)
	t.getElapsed()

	t = ExecutionTimer("[GMRF-W] Wind estimation")
	w.estimate()
	t.getElapsed()

	#t = ExecutionTimer("Wind uncertainty")
	#w.computeUncertainty()
	#t.getElapsed()
	tt.getElapsed()

	w.getWindEstimate().plot(vmax=1, interpol=3 )

