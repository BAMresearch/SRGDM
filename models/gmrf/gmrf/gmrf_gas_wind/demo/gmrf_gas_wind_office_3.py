if(__name__ == "__main__"):
	import sys
	sys.path.insert(0, "../../../..")
	from gdm.utils.report import getInstanceSize
	from gdm.gmrf import GMRF_Gas_Wind
	from gdm.common.environments.small_office_3 import small_office_3
	from gdm.utils.benchmarking import ExecutionTimer
	import random
	from gdm.common import Observation, ObstacleMap
	import numpy as np

	small_office_3.gas.normalize()

	observations = []
	for i in range(0,100):
		position = (random.uniform(0,10), random.uniform(0,10))
		observations += [small_office_3.getObservation(position)]


	ggw = GMRF_Gas_Wind(small_office_3.obstacles)
	ggw.addObservation(observations)

	t0 = ExecutionTimer("Gas-Wind TOTAL")
	t1 = ExecutionTimer("Gas-Wind estimation")
	ggw.estimate()
	t1.getElapsed()

	"""
	t2 = ExecutionTimer("Gas-Wind Uncertainty")
	ggw.computeUncertainty()
	t2.getElapsed()
	t0.getElapsed()
	print(getInstanceSize(observations))
	print(getInstanceSize(ggw))
	print("")
	"""
	ggw.getGasEstimate().plot()
	ggw.getWindEstimate().plot(vmax=1)
	#ggw.getGasUncertainty().plot(vmax=1)
	#ggw.toNormalDistributionMap().plot()

	#small_office_3.gas.plot(vmax=1)
	"""
	"""