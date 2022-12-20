if __name__ == "__main__":
	import sys
	sys.path.insert(0, "../../../..")
	from gdm.gmrf import GMRF_Wind, GMRF_Wind_Efficient
	from gdm.common.environments.small_office_3 import small_office_3
	from gdm.utils.benchmarking import ExecutionTimer
	import random
	from gdm.common import Observation, ObstacleMap
	import numpy as np


	observations = []
	for i in range(0,10):
		position = (random.uniform(0,10), random.uniform(0,10))
		observations += [small_office_3.getObservation(position)]


	observations = [Observation((0.5,5.5), wind=(1,0), data_type='wind'),
	                Observation((0.5,8.5), wind=(-1,0), data_type='wind')]


	w = GMRF_Wind_Efficient(small_office_3.obstacles, resolution=0.2)
	w.addObservation(observations)

	t = ExecutionTimer("Wind estimation")
	w.estimate()
	t.getElapsed()

	w.getWindEstimate().plot(vmax=1.0)
