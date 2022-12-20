import numpy as np
from gdm.common import Observation, ObstacleMap


m = np.array((
	(.9,.9,.9,.9),
	(.9,.1,.1,.9),
	(.9,.1,.1,.9),
	(.9,.1,.1,.9),
	(.9,.9,.9,.9),
))

obstacle_map = ObstacleMap.fromMatrix(m, resolution=1)

observations = [Observation((1.1, 1.1), gas=0.0, wind=(-1, 1), data_type='gas+wind'),
	            Observation((1.1, 2.0), gas=0.0, wind=( 1, 1), data_type='gas+wind'),
	            Observation((2.0, 2.0), gas=1.0, wind=( 1,-1), data_type='gas+wind'),
				Observation((2.0, 1.1), gas=1.0, wind=(-1,-1), data_type='gas+wind')]
