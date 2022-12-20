if(__name__ is "__main__"):

	from gdm.gmrf import GMRF_Gas_Wind_Efficient
	from gdm.common.environments.corridor_1 import corridor_1, corridor_1_ccm_100
	from gdm.utils.benchmarking import ExecutionTimer


	ggw = GMRF_Gas_Wind_Efficient(corridor_1.obstacles.toCellConnectivityMap(),
		sigma_gw=10)


	path = [(1,1), (1,2), (1,0.5), (2,1), (2,2), (2,0.5), (3,1), (3,2), (3,0.5), (4,1), (4,2), (4,0.5), (2.5, 3) , (2.5, 4)]
	observations = corridor_1.getPosition(path)
	ggw.addObservation(observations)

	t0 = ExecutionTimer("Gas-Wind TOTAL")
	t1 = ExecutionTimer("Gas-Wind estimation")
	ggw.estimate()
	t1.getElapsed()

	t2 = ExecutionTimer("Gas-Wind Uncertainty")
	#ggw.computeUncertainty()
	t2.getElapsed()
	t0.getElapsed()

	## PLOT
	ggw.getGasEstimate().plot()
	ggw.getWindEstimate().plot(interpol=1)
	ggw.getGasUncertainty().plot()
	#ggw.toNormalDistributionMap().plot()

