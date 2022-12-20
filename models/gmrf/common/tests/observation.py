if(__name__ is "__main__"):

	from ....gdm.common import Observation
	from ....gdm.utils.report import getInstanceSize
	import random

	o = []
	for i in range(0, 100):
		position = (random.uniform(0, 10), random.uniform(0, 10))
		gas = random.uniform(0, 10)
		wind = (random.uniform(0, 10), random.uniform(0, 10))
		o += [Observation(position=position, gas=gas, wind=wind, data_type='gas+wind')]
		print(getInstanceSize(o))