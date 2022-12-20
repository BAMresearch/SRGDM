def train(environment, wind_gt, ccm, inc=0.05):

	from gdm.common import ObstacleMap, Lattice2DScalar
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


	from gdm.gmrf import GMRF_Wind_Efficient
	import random
	from gdm.utils.metrics import getDNLML2

	sigma_wr = 22.507
	sigma_wc = 0.0052
	sigma_wo = 0.0311
	count = 0


	while count < 100:

		## GET RANDOM SEGMENTS WITH RANDOM START POSITION
		max_steps = 200
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
		print("Create estimators...")
		grd = GMRF_Wind_Efficient(ccm, sigma_wr=sigma_wr*(1-inc), sigma_wc=sigma_wc,         sigma_wo=sigma_wo)
		gru = GMRF_Wind_Efficient(ccm, sigma_wr=sigma_wr*(1+inc), sigma_wc=sigma_wc,         sigma_wo=sigma_wo)
		gcd = GMRF_Wind_Efficient(ccm, sigma_wr=sigma_wr,         sigma_wc=sigma_wc*(1-inc), sigma_wo=sigma_wo)
		gcu = GMRF_Wind_Efficient(ccm, sigma_wr=sigma_wr,         sigma_wc=sigma_wc*(1+inc), sigma_wo=sigma_wo)
		god = GMRF_Wind_Efficient(ccm, sigma_wr=sigma_wr,         sigma_wc=sigma_wc,         sigma_wo=sigma_wo*(1-inc))
		gou = GMRF_Wind_Efficient(ccm, sigma_wr=sigma_wr,         sigma_wc=sigma_wc,         sigma_wo=sigma_wo*(1+inc))
		
		print("Add observations and predict")
		grd.addObservation(observations).estimate()
		gru.addObservation(observations).estimate()
		gcd.addObservation(observations).estimate()
		gcu.addObservation(observations).estimate()
		god.addObservation(observations).estimate()
		gou.addObservation(observations).estimate()
		
		print("gt")
		wind_gt_0 = Lattice2DScalar.fromMatrix(wind_gt.toMatrix(0))
		wind_gt_1 = Lattice2DScalar.fromMatrix(wind_gt.toMatrix(1))
		
		print("To distributions")
		grd_mean_0 = Lattice2DScalar.fromMatrix(grd.getWindEstimate().toMatrix(0))
		grd_mean_1 = Lattice2DScalar.fromMatrix(grd.getWindEstimate().toMatrix(1))		
		gru_mean_0 = Lattice2DScalar.fromMatrix(gru.getWindEstimate().toMatrix(0))
		gru_mean_1 = Lattice2DScalar.fromMatrix(gru.getWindEstimate().toMatrix(1))
		
		gcd_mean_0 = Lattice2DScalar.fromMatrix(gcd.getWindEstimate().toMatrix(0))
		gcd_mean_1 = Lattice2DScalar.fromMatrix(gcd.getWindEstimate().toMatrix(1))		
		gcu_mean_0 = Lattice2DScalar.fromMatrix(gcu.getWindEstimate().toMatrix(0))
		gcu_mean_1 = Lattice2DScalar.fromMatrix(gcu.getWindEstimate().toMatrix(1))
		
		god_mean_0 = Lattice2DScalar.fromMatrix(god.getWindEstimate().toMatrix(0))
		god_mean_1 = Lattice2DScalar.fromMatrix(god.getWindEstimate().toMatrix(1))		
		gou_mean_0 = Lattice2DScalar.fromMatrix(gou.getWindEstimate().toMatrix(0))
		gou_mean_1 = Lattice2DScalar.fromMatrix(gou.getWindEstimate().toMatrix(1))
		
		print("Evaluate")
		mrd = getDNLML2(grd_mean_0, wind_gt_0) + getDNLML2(grd_mean_1, wind_gt_1)
		mru = getDNLML2(gru_mean_0, wind_gt_0) + getDNLML2(gru_mean_1, wind_gt_1)
		mcd = getDNLML2(gcd_mean_0, wind_gt_0) + getDNLML2(gcd_mean_1, wind_gt_1)
		mcu = getDNLML2(gcu_mean_0, wind_gt_0) + getDNLML2(gcu_mean_1, wind_gt_1)
		mod = getDNLML2(god_mean_0, wind_gt_0) + getDNLML2(god_mean_1, wind_gt_1)
		mou = getDNLML2(gou_mean_0, wind_gt_0) + getDNLML2(gou_mean_1, wind_gt_1)

		if mrd <= mru:
			sigma_wr = sigma_wr * (1-inc)
		else:
			sigma_wr = sigma_wr * (1+inc)
			
		if mcd <= mcu:
			sigma_wc = sigma_wc * (1-inc)
		else:
			sigma_wc = sigma_wc * (1+inc)
			
		if mod <= mou:
			sigma_wo = sigma_wo * (1-inc)
		else:
			sigma_wo = sigma_wo * (1+inc)

		
		grd.getWindEstimate().plot()

		## SUMMARY
		count += 1
		print("Count: " + str(count) + "\tsigma: " + str(sigma_wr) + "\t" + str(sigma_wc) + "\t" + str(sigma_wo) + "\tsteps: " + str(steps) + "\start: " +str(start))
		


if(__name__ is "__main__"):

	from gdm.common.environments import small_office_3_running as env_data
	environment = env_data.small_office_3
	wind_gt = environment.wind
	ccm = environment.obstacles.toCellConnectivityMap()

	train(environment, wind_gt, ccm, inc=0.015)