from gdm.common.environments.corridor_1_running import corridor_1
from gdm.common.environments.mapir_4b_running import mapir_4b
from gdm.gmrf import GMRF_Gas
from gdm.kdm import KDM_VW
from gdm.gmrf import GMRF_Gas_Wind_Efficient, GMRF_Gas_Efficient
from gdm.common import Lattice2DScalar, NormalDistributionMapper, Observation
import gdm.utils.metrics
import sys
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


max_gas=0.8


def compareWithGroundtruth(mapper, ground_truth, mask):
	assert isinstance(ground_truth, Lattice2DScalar)
	assert isinstance(mapper, NormalDistributionMapper)
	assert isinstance(mask, Lattice2DScalar)
	assert mapper.shape == mask.shape

	pdf = mapper.toNormalDistributionMap()
	mean = pdf.getMean()
	var = pdf.getVariance()

	varm = var.toMatrix()
	varm[mask.toMatrix()<=0.1] = 1.0
	var.loadMatrix(varm)
	#var.plot()

	error = gdm.utils.metrics.getError(mean, ground_truth)
	dist = gdm.utils.metrics.getDistance(mean, ground_truth)
	rmse = gdm.utils.metrics.getRMSE(mean, ground_truth)
	md = gdm.utils.metrics.getMahalanobisDistance(pdf, ground_truth)
	nlml = gdm.utils.metrics.getNLML(pdf, ground_truth)

	max_gas = mean.toMatrix().max()
	min_gas = mean.toMatrix().min()
	avg_gas = mean.toMatrix().sum() / (mean.shape[0] * mean.shape[1])

	max_var = var.toMatrix().max()
	min_var = var.toMatrix().min()
	avg_var = var.toMatrix().sum() / (var.shape[0] * var.shape[1])

	print("Error:\t" + str(error) + "\t" +
		  "Dist:\t" + str(dist) + "\t" +
		  "RSME:\t" + str(rmse) + "\t" +
		  "MDis:\t" + str(md) + "\t" +
		  "NLML:\t" + str(nlml) + "\t" +
		  "max_gas:\t" + str(max_gas) + "\t" +
		  "min_gas:\t" + str(min_gas) + "\t" +
		  "avg_gas:\t" + str(avg_gas) + "\t" +
		  "max_var:\t" + str(max_var) + "\t" +
		  "min_var:\t" + str(min_var) + "\t" +
		  "avg_var:\t" + str(avg_var) + "\t")

	return error, dist, rmse, md, nlml






def testNoise(name, environment, path_checkpoints, scale=False, proportional=False):

	## GET OBSERVATIONS
	path = []
	for i in range(0,len(path_checkpoints)-1):
		checkpoint_1 = path_checkpoints[i]
		checkpoint_2 = path_checkpoints[i+1]
		path += environment.getStraightPathBetweenPositions(checkpoint_1, checkpoint_2)
	observations = []
	inc = 10
	for i in range(1, len(path)):
		position = path[i]
		obs = environment.getObservation(position, i/inc)
		assert -10 < obs.wind[0] < 10
		assert -10 < obs.wind[1] < 10
		observations += [obs]


	ccm = environment.obstacles.toCellConnectivityMap()
	mask = environment.obstacles.invertValues()
	inc = 30

	R = []
	S = []
	GDM = []
	RMSE = []
	NLML = []
	SCENARIO = []

	sigmas = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1)
	for sigma in sigmas:

		scale_factor=1
		if scale:
			scale_factor = max(sigma/0.1, 1) #aumentar sigma_zg etc en cuanto el ruido sube de 0.1

		for r in range(0,25):

			print("-----------------")
			print(name)
			print(r)
			print(sigma)

			noisy_observations = []
			if proportional:
				for observation in observations:
					noisy_gas = max(0.0, np.random.normal(observation.gas, np.abs(0.001+observation.gas*sigma)))
					noisy_wind = (	np.random.normal(observation.wind[0], np.abs(0.001+observation.wind[0]*sigma)), np.random.normal(observation.wind[1], np.abs(0.001+observation.wind[1]*sigma)))
					position = observation.position
					noisy_obs = Observation(position=position, gas=noisy_gas, wind=noisy_wind, data_type='gas+wind')
					noisy_observations += [noisy_obs]
			else:
				for observation in observations:
					noisy_gas = max(0, np.random.normal(observation.gas, sigma))
					noisy_wind = (np.random.normal(observation.wind[0], sigma), np.random.normal(observation.wind[1], sigma))
					position = observation.position
					noisy_obs = Observation(position=position, gas=noisy_gas, wind=noisy_wind, data_type='gas+wind')
					noisy_observations += [noisy_obs]


			gmrfg  = GMRF_Gas_Efficient(ccm, sigma_gz=scale_factor*1)
			gmrfg.addObservation(noisy_observations)
			gmrfg.getGasEstimate()
			gmrfg.getGasUncertainty()
			if r == 0:
				save = "/home/andy/tmp/gmrf/" + "g-gmrf_noise_" + name + "_" + str(sigma) + "_" + str(r)
				gmrfg.getGasEstimate().plot(mask=1, vmax=max_gas, save=save+"_mean.png")
				gmrfg.getGasUncertainty().plot(mask=1, vmax=1,    save=save+"_sigma.png")
			error, dist, rmse, md, nlml = compareWithGroundtruth(gmrfg, environment.gas, mask)
			SCENARIO += [name]
			R += [r]
			S += [sigma]
			GDM += ["G-GMRF"]
			RMSE += [rmse]
			NLML += [nlml]

			kdmvw = KDM_VW(ccm, scaling_std=scale_factor*4.956)
			kdmvw.addObservation(noisy_observations)
			kdmvw.getGasEstimate()
			kdmvw.getGasUncertainty()
			if r == 0:
				save = "/home/andy/tmp/gmrf/" + "kdm_noise_" + name + "_" + str(sigma) + "_" + str(r)
				kdmvw.getGasEstimate().plot(mask=1, vmax=max_gas, save=save+"_mean.png")
				kdmvw.getGasUncertainty().plot(mask=1, vmax=1,    save=save+"_sigma.png")
			error, dist, rmse, md, nlml = compareWithGroundtruth(kdmvw, environment.gas, mask)
			SCENARIO += [name]
			R += [r]
			S += [sigma]
			GDM += ["KDM+V/W"]
			RMSE += [rmse]
			NLML += [0]

			gmrfgw = GMRF_Gas_Wind_Efficient(ccm, sigma_gz=scale_factor*1, sigma_wz=scale_factor*0.1)
			gmrfgw.addObservation(noisy_observations)
			gmrfgw.getGasEstimate()
			gmrfgw.getGasUncertainty()
			if r == 0:
				save = "/home/andy/tmp/gmrf/" + "gw-gmrf_noise_" + name + "_" + str(sigma) + "_" + str(r)
				gmrfgw.getGasEstimate().plot(mask=1, vmax=max_gas, save=save+"_mean.png")
				gmrfgw.getWindEstimate().plot(mask=1, vmax=1,      save=save+"_wind.png")
				gmrfgw.getGasUncertainty().plot(mask=1, vmax=1,    save=save+"_sigma.png")
			error, dist, rmse, md, nlml = compareWithGroundtruth(gmrfgw, environment.gas, mask)
			SCENARIO += [name]
			R += [r]
			S += [sigma]
			GDM += ["GW-GMRF"]
			RMSE += [rmse]
			NLML += [nlml]
			
			assert id(gmrfg) != id(gmrfgw)
			assert(noisy_observations[0].hasWind())
			assert(len(noisy_observations) == len(observations))
			assert(len(RMSE)==len(GDM))


	data = pandas.DataFrame({'scenario' : SCENARIO,
							 'run': R,
							 'sigma': S,
							 'gdm' : GDM,
							 'rmse': RMSE,
							 'nlml' : NLML})


	data.to_csv('/home/andy/tmp/noise_comparison_'+ name +'.csv')




######################################################################################################
"""
name = "corridor"
environment = corridor_1
path_checkpoints = ((0.5,3.5),(14.5,3.5), (14.5, 4.5), (0.5,4.5))
testNoise(name, environment, path_checkpoints)

name = "corridor_full"
environment = corridor_1
path_checkpoints = ((0.5,4),(2.0,4),(2.2,3.8),(2.2,2.4),(2,2),(1,2),(0.8,1.8),(0.8,0.8), (1,0.8), (4,0.8), (4.2,1), (4.2,2), (4,2.2), (2.8,2.2), (2.6, 3.8), (3,4), (5,4.2),(8.8,4), (9,3.8), (9.1,2),(8.8,1.8),(6,1.8),(5.8,1.6),(5.8,1), (6,0.8),(9,0.8), (9.3,1), (9.3,3), (10,3), (10.2,2.8),(10.3,1),(10.9,0.8),(13.5,0.8),(14,1),(14,2),(13.8,2.2),(10.8,2.6),(10.6,3),(10.6,3.8),(10.8,4), (14.3,4), (14.3,4.5), (12,3.3), (10,4.5), (8,3.3), (6,4.5), (4,3.3),(2,4.5), (0.5,3.3))
testNoise(name, environment, path_checkpoints)

name = "lab4"
environment = mapir_4b
path_checkpoints = ((0.7,10.5),(1,10), (2.5,4), (2.5,1), (2.5,1), (2.7,3),  (2.5,3), (1,3.2), (4.5,3.3), (2.5,3.5),(3.5,8.5), (3.5,8.5), (6,9.5), (7.2,6.5), (9,6), (9,6), (9.5,4), (5.6,3), (7.5,3), (7.5,1.5), (8,3), (9.5, 3.5), (9.5, 3.5), (9.5,6), (8.5,7), (8.5,10.5), (8.5,10.5), (0.7,10.5))
testNoise(name, environment, path_checkpoints)
"""
"""
name = "corridor_scaled"
environment = corridor_1
path_checkpoints = ((0.5,3.5),(14.5,3.5), (14.5, 4.5), (0.5,4.5))
testNoise(name, environment, path_checkpoints, True)

name = "corridor_full_scaled"
environment = corridor_1
path_checkpoints = ((0.5,4),(2.0,4),(2.2,3.8),(2.2,2.4),(2,2),(1,2),(0.8,1.8),(0.8,0.8), (1,0.8), (4,0.8), (4.2,1), (4.2,2), (4,2.2), (2.8,2.2), (2.6, 3.8), (3,4), (5,4.2),(8.8,4), (9,3.8), (9.1,2),(8.8,1.8),(6,1.8),(5.8,1.6),(5.8,1), (6,0.8),(9,0.8), (9.3,1), (9.3,3), (10,3), (10.2,2.8),(10.3,1),(10.9,0.8),(13.5,0.8),(14,1),(14,2),(13.8,2.2),(10.8,2.6),(10.6,3),(10.6,3.8),(10.8,4), (14.3,4), (14.3,4.5), (12,3.3), (10,4.5), (8,3.3), (6,4.5), (4,3.3),(2,4.5), (0.5,3.3))
testNoise(name, environment, path_checkpoints, True)

name = "lab4_scaled"
environment = mapir_4b
path_checkpoints = ((0.7,10.5),(1,10), (2.5,4), (2.5,1), (2.5,1), (2.7,3),  (2.5,3), (1,3.2), (4.5,3.3), (2.5,3.5),(3.5,8.5), (3.5,8.5), (6,9.5), (7.2,6.5), (9,6), (9,6), (9.5,4), (5.6,3), (7.5,3), (7.5,1.5), (8,3), (9.5, 3.5), (9.5, 3.5), (9.5,6), (8.5,7), (8.5,10.5), (8.5,10.5), (0.7,10.5))
testNoise(name, environment, path_checkpoints, True)
"""

"""
name = "corridor_prop"
environment = corridor_1
path_checkpoints = ((0.5,3.5),(14.5,3.5), (14.5, 4.5), (0.5,4.5))
testNoise(name, environment, path_checkpoints, proportional=True)

name = "corridor_full_prop"
environment = corridor_1
path_checkpoints = ((0.5,4),(2.0,4),(2.2,3.8),(2.2,2.4),(2,2),(1,2),(0.8,1.8),(0.8,0.8), (1,0.8), (4,0.8), (4.2,1), (4.2,2), (4,2.2), (2.8,2.2), (2.6, 3.8), (3,4), (5,4.2),(8.8,4), (9,3.8), (9.1,2),(8.8,1.8),(6,1.8),(5.8,1.6),(5.8,1), (6,0.8),(9,0.8), (9.3,1), (9.3,3), (10,3), (10.2,2.8),(10.3,1),(10.9,0.8),(13.5,0.8),(14,1),(14,2),(13.8,2.2),(10.8,2.6),(10.6,3),(10.6,3.8),(10.8,4), (14.3,4), (14.3,4.5), (12,3.3), (10,4.5), (8,3.3), (6,4.5), (4,3.3),(2,4.5), (0.5,3.3))
testNoise(name, environment, path_checkpoints, proportional=True)

name = "lab4_prop"
environment = mapir_4b
path_checkpoints = ((0.7,10.5),(1,10), (2.5,4), (2.5,1), (2.5,1), (2.7,3),  (2.5,3), (1,3.2), (4.5,3.3), (2.5,3.5),(3.5,8.5), (3.5,8.5), (6,9.5), (7.2,6.5), (9,6), (9,6), (9.5,4), (5.6,3), (7.5,3), (7.5,1.5), (8,3), (9.5, 3.5), (9.5, 3.5), (9.5,6), (8.5,7), (8.5,10.5), (8.5,10.5), (0.7,10.5))
testNoise(name, environment, path_checkpoints, proportional=True)
"""


def noise_boxplot(file):
	data = pandas.read_csv(file)
	sns.set_style("whitegrid", {"font.family":"Times New Roman"})
	#sns.set(font='times-new-roman')
	data = data[data.sigma!= 0.05]
	data = data[data.sigma!= 0.1]
	fig, ax = plt.subplots(figsize=(10,5))
	plt.ylim(0, 0.6)	
	#ax = sns.pointplot(x='sigma', y='rmse', hue='gdm',  data=data.groupby(['sigma', 'gdm'], as_index=False).mean())
	ax = sns.boxplot(x='sigma', y='rmse', hue='gdm', data=data, whis=10, width=0.8, ax=ax,linewidth=1, palette=sns.color_palette(["#00a933", "#ff420e", "#2a6099"]))
	#ax = sns.pointplot(x='sigma', y='rmse', hue='gdm', data=data, ax=ax, palette=sns.color_palette(["#00a933", "#ff420e", "#2a6099"]))
	plt.savefig(file+"_rmse.svg", format="svg")
	
	sns.set_style("whitegrid", {"font.family":"Times New Roman"})
	data = data[data.gdm!= "KDM-V/W"]
	fig, ax = plt.subplots(figsize=(10,5))
	ax = sns.boxplot(x='sigma', y='nlml', hue='gdm', data=data, whis=10, width=0.8, ax=ax,linewidth=1, palette=sns.color_palette(["#00a933", "#2a6099"]))
	plt.savefig(file+"_nlml.svg", format="svg")


def noise_lineplot(file, scenario):
	data = pandas.read_csv(file)
	size = (3.227, 3.25)
	font_size = 12
	sns.set_style("whitegrid", {"font.family": "Times New Roman"})
	plt.rcParams["xtick.labelsize"] = font_size
	plt.rcParams["ytick.labelsize"] = font_size
	plt.rcParams["legend.fontsize"] = font_size
	plt.rcParams['figure.dpi'] = 300
	fig, ax = plt.subplots(figsize=size)

	x = "SD of sensor noise"
	data.rename(columns={"sigma": x, "nlml": "NLML", "rmse": "RMSE"}, inplace=True)

	fig, ax = plt.subplots(figsize=size)
	plt.ylim(0, 0.5)
	ax = sns.lineplot(x=x, y='RMSE', hue='gdm', data=data, ax=ax, linewidth=1, palette=sns.color_palette(["#00a933", "#ff420e", "#2a6099"]))
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:])
	plt.savefig(file + "_rmse.svg", format="svg")


	data = data[data.gdm != "KDM+V/W"]
	fig, ax = plt.subplots(figsize=size)
	plt.ylim(0, 70000)
	ax = sns.lineplot(x=x, y='NLML', hue='gdm', data=data, ax=ax, linewidth=1, palette=sns.color_palette(["#00a933", "#2a6099"]))
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:])
	plt.savefig(file +"_nlml.svg", format="svg")


"""
name = "lab4"
environment = mapir_4b
path_checkpoints = ((0.7,10.5),(1,10), (2.5,4), (2.5,1), (2.5,1), (2.7,3),  (2.5,3), (1,3.2), (4.5,3.3), (2.5,3.5),(3.5,8.5), (3.5,8.5), (6,9.5), (7.2,6.5), (9,6), (9,6), (9.5,4), (5.6,3), (7.5,3), (7.5,1.5), (8,3), (9.5, 3.5), (9.5, 3.5), (9.5,6), (8.5,7), (8.5,10.5), (8.5,10.5), (0.7,10.5))
testNoise(name, environment, path_checkpoints, scale=False)

name = "lab4_compensated"
environment = mapir_4b
path_checkpoints = ((0.7,10.5),(1,10), (2.5,4), (2.5,1), (2.5,1), (2.7,3),  (2.5,3), (1,3.2), (4.5,3.3), (2.5,3.5),(3.5,8.5), (3.5,8.5), (6,9.5), (7.2,6.5), (9,6), (9,6), (9.5,4), (5.6,3), (7.5,3), (7.5,1.5), (8,3), (9.5, 3.5), (9.5, 3.5), (9.5,6), (8.5,7), (8.5,10.5), (8.5,10.5), (0.7,10.5))
testNoise(name, environment, path_checkpoints, scale=True)
"""

noise_lineplot('/home/andy/tmp/noise_comparison_lab4_compensated.csv', "lab4")
noise_lineplot('/home/andy/tmp/noise_comparison_lab4.csv', "lab4")
