if(__name__ == "__main__"):
	import sys
	sys.path.insert(0, "../../../..")
	from gdm.utils.report import getInstanceSize
	from gdm.gmrf import GMRF_Gas_Wind_Efficient
	from gdm.common.environments.wind_tunnel_short import wind_tunnel_short
	from gdm.utils.benchmarking import ExecutionTimer
	import random
	from gdm.common import Observation, ObstacleMap
	import numpy as np

	obstacle_map = wind_tunnel_short
	#wind_tunnel_short.gas.normalize()

	observations = []
	for i in range (0,5):
		position=(1,2.5-i*1/7)
		wind_x=np.random.normal(-1,0.25)
		wind_y = np.random.normal(0, 0.25)
		wind=(wind_x,wind_y)
		observations += [Observation(position=position, wind=wind, data_type="gas+wind")]
	for i in range (0,5):
		position=(1,2.5+i*2.5/5)
		wind_x=np.random.normal(-1,0.25)
		wind_y = np.random.normal(0, 0.25)
		wind=(wind_x,wind_y)
		observations += [Observation(position=position, wind=wind, data_type="gas+wind")]
	for i in range (0,4):
		position=(1+i/3,4.8)
		wind_x=np.random.normal(-1,0.2)
		wind_y = np.random.normal(0, 0.2)
		wind=(wind_x,wind_y)
		observations += [Observation(position=position, wind=wind, data_type="gas+wind")]

	ggw = GMRF_Gas_Wind_Efficient(obstacle_map.toCellConnectivityMap())
	ggw.addObservation(observations)

	t0 = ExecutionTimer("Gas-Wind TOTAL")
	t1 = ExecutionTimer("Gas-Wind estimation")
	ggw.estimate()
	t1.getElapsed()

	t2 = ExecutionTimer("Gas-Wind Uncertainty")
	ggw.computeUncertainty()
	t2.getElapsed()
	t0.getElapsed()
	print(getInstanceSize(observations))
	print(getInstanceSize(ggw))
	print("")

	ggw.getGasEstimate().plot()
	ggw.getWindEstimate().plot(vmax=1.1, interpol=3)
	ggw.getGasUncertainty().plot(vmax=1)
	#ggw.toNormalDistributionMap().plot()

	#small_office_3.gas.plot(vmax=1)
