import sys
sys.path.insert(0, "../../../..")







if(__name__ is "__main__"):

	from gdm.gmrf import GMRF_Gas_Wind, GMRF_Gas_Wind_Efficient
	from gdm.common.debug_setups.debug_setup_2 import obstacle_map, observations
	from gdm.utils.benchmarking import ExecutionTimer

	ggw = GMRF_Gas_Wind_Efficient(obstacle_map.toCellConnectivityMap())
	ggw.addObservation(observations)

	t0 = ExecutionTimer("Gas-Wind TOTAL")
	t1 = ExecutionTimer("Gas-Wind estimation")
	#ggw.estimate()
	t1.getElapsed()

	t2 = ExecutionTimer("Gas-Wind Uncertainty")
	#ggw.computeUncertainty()
	t2.getElapsed()
	t0.getElapsed()

	## PLOT
	#ggw.getGasEstimate().plot()
	#ggw.getWindEstimate().plot()
	#ggw.getGasUncertainty().plot()
	#ggw.toNormalDistributionMap().plot()



	import random
	from gdm.common.environments.corridor_1 import corridor_1
	from gdm.utils.metrics.nlml import getNLML

	ggw = GMRF_Gas_Wind_Efficient(corridor_1.obstacles.toCellConnectivityMap())


	observations = []
	for i in range(0, 10000):
		position = (random.uniform(0, 15), random.uniform(0, 5))
		observations += [corridor_1.getObservation(position)]
	ggw.addObservation(observations)
	ggw.estimate()

	print(getNLML(ggw.toNormalDistributionMap(), corridor_1.gas))