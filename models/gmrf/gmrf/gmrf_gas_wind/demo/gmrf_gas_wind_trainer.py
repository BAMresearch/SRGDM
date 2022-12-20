def train(environment, gas_gt, ccm, start_sigma, inc=0.05):

	from gdm.common import ObstacleMap
	import numpy as np
	import math


	def advanceUntilColission(obstacle_map, start_position, step_size=1, iterations=500):
		assert isinstance(obstacle_map, ObstacleMap)

		path_segments = []
		current_position = start_position
		current_angle = np.random.uniform(-np.pi, np.pi)

		while len(path_segments) < iterations:
			move_vector = (step_size * math.cos(current_angle),
			               step_size * math.sin(current_angle))
			path_segment = obstacle_map.getFeasiblePathForVector(
				current_position, move_vector)
			new_position = path_segment[-1]

			moved = math.sqrt((new_position[0] - current_position[0]) ** 2 + (
						new_position[1] - current_position[1]) ** 2)
			if moved < 0.9 * step_size:
				current_angle = np.random.uniform(-np.pi, np.pi)

			path_segments += [path_segment]
			current_position = new_position

		return path_segments


	from gdm.gmrf import GMRF_Gas_Wind_Efficient
	import random
	from gdm.utils.metrics import getNLML, getMahalanobisDistance, getDNLML, getDNLML2
	from gdm.utils.metrics.basic import getRMSE, testMetric

	sigma_gw = start_sigma
	sigma_gw_acc = 0
	count = 0


	while count < 100:

		## GET RANDOM SEGMENTS WITH RANDOM START POSITION
		max_steps = 1000
		steps = random.uniform(10, max_steps)
		has_obstacles = True
		start = (2.5, 2.5)
		while has_obstacles:
			start_x = random.uniform(0, environment.size[0])
			start_y = random.uniform(0, environment.size[1])
			start = (start_x, start_y)
			has_obstacles = environment.obstacles.hasObstacleAtPosition(start)
		path_segments = advanceUntilColission(environment.obstacles, start, iterations=steps)


		## GET OBSERVATIONS
		observations = []
		n = len(path_segments)
		for i in range(0,n):
			path_segment = path_segments[i]
			observations += environment.getObservation(path_segment, time=i)


		## ALTERNATIVES
		gd = GMRF_Gas_Wind_Efficient(ccm, sigma_gw=sigma_gw*(1-inc))
		gu = GMRF_Gas_Wind_Efficient(ccm, sigma_gw=sigma_gw*(1+inc))
		gd.addObservation(observations)
		gu.addObservation(observations)

		md = getDNLML2(gd.getGasEstimate(), gas_gt)
		mu = getDNLML2(gu.getGasEstimate(), gas_gt)
		m = 9999999
		if md <= mu:
			m = md
			sigma_gw = sigma_gw * (1-inc)
		elif md > mu:
			m = mu
			sigma_gw = sigma_gw * (1+inc)


		gd.getGasEstimate().plot(mask=1, vmax=1)
		#gd.getGasUncertainty().plot(vmax=1)
		#gas_gt.plot(mask=1)

		## SUMMARY
		count += 1
		sigma_gw_acc += sigma_gw
		print("Count: " + str(count) + "\tsigma: " + str(sigma_gw) + "\tm: " + str(m) + "\tavg sigma: " + str(sigma_gw_acc/count) + "\tsteps: " + str(steps) + "\start: " +str(start))



if(__name__ is "__main__"):

	from gdm.common.environments import small_office_3_running as env_data
	environment = env_data.small_office_3
	gas_gt = environment.gas
	ccm = environment.obstacles.toCellConnectivityMap()

	train(environment, gas_gt, ccm, start_sigma=0.028, inc=0.05)