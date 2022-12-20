if(__name__ is "__main__"):

	from gdm.gmrf import GMRF_Wind_Efficient, GMRF_Wind
	from gdm.common.environments.corridor_1 import corridor_1, corridor_1_ccm_100, \
	corridor_1_obstacles_025, 	corridor_1_obstacles_050
	from gdm.utils.benchmarking import ExecutionTimer


	w = GMRF_Wind_Efficient(corridor_1_obstacles_050.toCellConnectivityMap())


	path = [(1,1), (1,2), (1,0.5), (2,1), (2,2), (2,0.5), (3,1), (3,2), (3,0.5), (4,1), (4,2), (4,0.5), (2.5, 3) , (2.5, 4)]
	observations = corridor_1.getPosition(path)
	w.addObservation(observations)

	t0 = ExecutionTimer("Gas-Wind TOTAL")
	t1 = ExecutionTimer("Gas-Wind estimation")
	w.getWindEstimate()
	t1.getElapsed()

	## PLOT
	w.getWindEstimate().plot(interpol=1)
	#w.getWindUncertainty().plot()


