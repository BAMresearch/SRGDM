from gdm.common.environments.corridor_1_running import corridor_1
from gdm.common.environments.mapir_4b_running import mapir_4b
from gdm.gmrf import GMRF_Gas
from gdm.kdm import KDM_VW
from gdm.gmrf import GMRF_Gas_Wind_Efficient, GMRF_Gas_Efficient
from gdm.common import Lattice2DScalar, NormalDistributionMapper
import gdm.utils.metrics
import sys
import numpy as np
import pandas
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


#environment = corridor_1
environment = mapir_4b
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





def test():
	
	ccm = environment.obstacles.toCellConnectivityMap()
	mask = environment.obstacles.invertValues()
	inc = 30
	
	
	R = []
	N = []
	GDM = []
	RMSE = []
	NLML = []


	num_samples = [1, 10, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000]
	for n in num_samples:
	
		for r in range(0,25):
	
			print(n)
			print(r)
	
			observations = []
			max_wind = 0
			for i in range(0, n):
				position = (random.uniform(0, environment.size[0]), random.uniform(0, environment.size[1]))
				obs = environment.getObservation(position, i/inc)
				max_wind = max(max_wind, obs.wind[0])
				observations += [obs]
			assert observations[0].data_type == "2"
			print(max_wind)
	
	
			gmrfg = GMRF_Gas_Efficient(ccm)
			gmrfg.addObservation(observations)
			gmrfg.getGasEstimate()
			gmrfg.getGasUncertainty()
			if r== -1:
				gmrfg.getGasEstimate().plot(save="/home/andy/tmp/gmrf/" + "g-gmrf_samples_mean")
				gmrfg.getGasUncertainty().plot(save="/home/andy/tmp/gmrf/" + "g-gmrf_samples_sigma")
			error, dist, rmse, md, nlml = compareWithGroundtruth(gmrfg, environment.gas, mask)
			R += [r]
			N += [n]
			GDM += ["G-GMRF"]
			RMSE += [rmse]
			NLML += [nlml]
	
	
			kdmvw = KDM_VW(ccm)
			kdmvw.addObservation(observations)
			kdmvw.getGasEstimate()
			if r== -1:
				kdmvw.getGasEstimate().plot(save="/home/andy/tmp/gmrf/" + "kdm_samples_mean")
				kdmvw.getGasUncertainty().plot(save="/home/andy/tmp/gmrf/" + "kdm_samples_sigma")
			error, dist, rmse, md, nlml = compareWithGroundtruth(kdmvw, environment.gas, mask)
			R += [r]
			N += [n]
			GDM += ["KDM-V/W"]
			RMSE += [rmse]
			NLML += [0]
	
	
	
			gmrfgw = GMRF_Gas_Wind_Efficient(ccm)
			gmrfgw.addObservation(observations)
			gmrfgw.getGasEstimate()
			gmrfgw.getWindEstimate()
			if r== -1:
				gmrfgw.getGasEstimate().plot(save="/home/andy/tmp/gmrf/" + "gw-gmrf_samples_mean")
				gmrfgw.getWindEstimate().plot(save="/home/andy/tmp/gmrf/" + "gw-gmrf_samples_wind")
			gmrfgw.getGasUncertainty().plot(save="/home/andy/tmp/gmrf/" + "gw-gmrf_samples_sigma")
			error, dist, rmse, md, nlml = compareWithGroundtruth(gmrfgw, environment.gas, mask)
			R += [r]
			N += [n]
			GDM += ["GW-GMRF"]
			RMSE += [rmse]
			NLML += [nlml]
	
	
	
	data = pandas.DataFrame({'GDM' : GDM,
							 'run': R,
							 'N': N,
							 'RMSE': RMSE,
							 'NLML': NLML})
	
	
	data.to_csv('/home/andy/tmp/num_samples_lab4.csv')
	
	
	
def plot():
	
	data = pandas.read_csv('/home/andy/tmp/num_samples_lab4.csv')
	data = data[data['N'] <= 5000]

	size = (7, 3.5)
	font_size = 12
	sns.set_style("whitegrid", {"font.family": "Times New Roman"})
	plt.rcParams["xtick.labelsize"] = font_size
	plt.rcParams["ytick.labelsize"] = font_size
	plt.rcParams["legend.fontsize"] = font_size
	plt.rcParams['figure.dpi'] = 300

	fig, ax = plt.subplots(figsize=size)
	#ax.set(xscale="log")
	sns.lineplot(x='N', y='RMSE', hue='GDM',  data=data, palette=sns.color_palette(["#00a933", "#ff420e", "#2a6099" ]))
	#handles, labels = ax.get_legend_handles_labels()
	#ax.legend(handles=handles[1:], labels=labels[1:])
	plt.savefig("/home/andy/tmp/num_samples_lab4_rsme.svg", format="svg")


	fig, ax = plt.subplots(figsize=size)
	data = data[data['GDM'] != "KDM-V/W"] 
	plt.ylim(-20000, 40000)
	sns.lineplot(x='N', y='NLML', hue='GDM',  data=data, palette=sns.color_palette(["#00a933", "#2a6099" ]))
	#handles, labels = ax.get_legend_handles_labels()
	#ax.legend(handles=handles[1:], labels=labels[1:])
	plt.savefig("/home/andy/tmp/num_samples_lab4_nlml.svg", format="svg")
	
	
#test()
plot()
