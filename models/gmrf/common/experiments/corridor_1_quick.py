import sys
sys.path.append("/home/nicolas/git/gdm/")

from common.environments.corridor_1_running import corridor_1
from common.environments.mapir_4b_running import mapir_4b
from gmrf import GMRF_Gas
from kdm import KDM_VW
from gmrf import GMRF_Gas_Wind_Efficient, GMRF_Gas_Efficient
from common import Lattice2DScalar, NormalDistributionMapper
import utils.metrics
import sys
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

## IMPORT CORRIDOR 1 (TIME-EVOLVING VERSION)
#corridor_1.plot()
#print(len(corridor_1.environments))
#corridor_1.environments[0].plot()



## DEFINE EXPLORATION PATH
cor_zz_1 = ((0.5,4.5),(2.0,3.5),(2.4,3.2),(2.2,1.8),(1.8,2.5),(0.5,2.5))
cor_zz_2 = ((0.5,2.5),(1.5,1.5),(0.5,1),(0.8,0.5), (2,1), (2.5,0.4), (3,1), (4.5,0.5), (3.5,1.5), (4.5,2.5), (2.8,1.8), (2.8,2.2),(2.6, 3), (3.8,4.5),(6,3.3))
cor_zz_3 = ((6,3.3),(7.5,4.5), (9.4,3.3),(9.1,1.8),(8.5,2.5),(7.5,1.5), (7,2.5), (6,1.5),(5.5,2.5))
cor_zz_4 = ((5.5,2.5), (5.6,0.5), (6.5,1.5), (7.5,0.5), (8.5,1.5), (9.5,0.5),(9.5,3.3), (10,3.5), (10.5,3.3), (10.3,2.5),(11,1.5))
cor_zz_5 = ((10.4,0.5), (11.5,1), (12.5,0.5), (13.5,1), (14.5,0.5), (13.5,1.5), (14.5,2.5), (12.8,1.5), (12,2.5), (11.5,1.5),(10.6,3),(12,4.5),(14.5,3.3),(14.5,4.5))
cor_zz_6 = ((14.5,4.5), (12,3.3), (10,4.5), (8,3.3), (6,4.5), (4,3.3),(2,4.5), (0.5,3.3), (0.5,4.5))

path_checkpoints = ((0.5,3.5),(14.5,3.5), (14.5, 4.5), (0.5,4.5))
#path_checkpoints = ((0.5,4),(2.0,4),(2.2,3.8),(2.2,2.4),(2,2),(1,2),(0.8,1.8),(0.8,0.8), (1,0.8), (4,0.8), (4.2,1), (4.2,2), (4,2.2), (2.8,2.2), (2.6, 3.8), (3,4), (5,4.2),(8.8,4), (9,3.8), (9.1,2),(8.8,1.8),(6,1.8),(5.8,1.6),(5.8,1), (6,0.8),(9,0.8), (9.3,1), (9.3,3), (10,3), (10.2,2.8),(10.3,1),(10.9,0.8),(13.5,0.8),(14,1),(14,2),(13.8,2.2),(10.8,2.6),(10.6,3),(10.6,3.8),(10.8,4), (14.3,4), (14.3,4.5), (12,3.3), (10,4.5), (8,3.3), (6,4.5), (4,3.3),(2,4.5), (0.5,3.3))
#path_checkpoints = cor_zz_1+cor_zz_2+cor_zz_3+cor_zz_4+cor_zz_5+cor_zz_6
#path_checkpoints = ((0.7,10.5),(1,10), (2.5,4), (2.5,1), (2.5,1), (2.7,3),  (2.5,3), (1,3.2), (4.5,3.3), (2.5,3.5),(3.5,8.5), (3.5,8.5), (6,9.5), (7.2,6.5), (9,6), (9,6), (9.5,4), (5.6,3), (7.5,3), (7.5,1.5), (8,3), (9.5, 3.5), (9.5, 3.5), (9.5,6), (8.5,7), (8.5,10.5), (8.5,10.5), (0.7,10.5))


environment = corridor_1
#environment = mapir_4b
max_gas=0.8

scenario = "corridor_quick"


path = []
for i in range(0,len(path_checkpoints)-1):
	checkpoint_1 = path_checkpoints[i]
	checkpoint_2 = path_checkpoints[i+1]
	path += environment.getStraightPathBetweenPositions(checkpoint_1, checkpoint_2)


def getPathLength(obs):
	length = 0
	for i in range(0,len(obs)-1):
		pos1 = obs[i].position
		pos2 = obs[i+1].position
		segment = np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
		length += segment
	#print("Length:\t" + str(length))
	return length


def compareWithGroundtruth(mapper, ground_truth, mask, observations):
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

	path_length = np.round(getPathLength(observations), 3)

	print("Path:\t" + str(path_length) + "\t" +
		  "Error:\t" + str(error) + "\t" +
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

	return path_length, rmse, nlml

def test():
	ccm = environment.obstacles.toCellConnectivityMap()
	mask = environment.obstacles.invertValues()

	## GET OBSERVATIONS
	observations = []
	inc = 1
	for i in range(1, len(path)):
		position = path[i]
		observations += [environment.getObservation(position, i/inc)]


	SCENARIO = []
	GDM = []
	PATH = []
	METRIC = []
	VALUE = []



	#sys.stdout = open('/home/andy/gmrfg.txt', 'w')
	for i in range(1, len(observations), inc):
		partial_observations = observations[0:i]
		gmrfg  = GMRF_Gas_Efficient(ccm)
		gmrfg.addObservation(partial_observations)
		gmrfg.estimate()
		gmrfg.getGasEstimate()#.plot(mask=1, vmax=max_gas, save="/home/andy/tmp/gmrf/" + "g-gmrf_" + str(i) + "_mean")
		gmrfg.getGasUncertainty()#.plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "g-gmrf_" + str(i) + "_sigma")
		path_length, rmse, nlml = compareWithGroundtruth(gmrfg, environment.gas, mask, partial_observations)
		SCENARIO += [scenario]
		GDM += ["G-GMRF"]
		PATH += [path_length]
		METRIC += ["RMSE"]
		VALUE += [rmse]
		SCENARIO += [scenario]
		GDM += ["G-GMRF"]
		PATH += [path_length]
		METRIC += ["NLML"]
		VALUE += [nlml]



	#sys.stdout = open('/home/andy/kdmvw.txt', 'w')
	for i in range(1, len(observations), inc):
		partial_observations = observations[0:i]

		kdmvw = KDM_VW(ccm)
		kdmvw.addObservation(partial_observations)
		kdmvw.estimate()
		kdmvw.getGasEstimate()#.plot(mask=1, vmax=max_gas, save="/home/andy/tmp/gmrf/" + "kdm_" + str(i) + "_mean")
		kdmvw.getGasUncertainty()#.plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "kdm_" + str(i) + "_sigma")
		path_length, rmse, nlml = compareWithGroundtruth(kdmvw, environment.gas, mask, partial_observations)
		SCENARIO += [scenario]
		GDM += ["KDM+V/W"]
		PATH += [path_length]
		METRIC += ["RMSE"]
		VALUE += [rmse]
		SCENARIO += [scenario]
		GDM += ["KDM+V/W"]
		PATH += [path_length]
		METRIC += ["NLML"]
		VALUE += [nlml]


	#sys.stdout = open('/home/andy/gmrfgw.txt', 'w')
	for i in range(1, len(observations), inc):
		partial_observations = observations[0:i]

		gmrfgw = GMRF_Gas_Wind_Efficient(ccm)
		gmrfgw.addObservation(partial_observations)
		gmrfgw.estimate()
		gmrfgw.getGasEstimate()#.plot(mask=1, vmax=max_gas, save="/home/andy/tmp/gmrf/" + "gw-gmrf_" + str(i) + "_mean")
		gmrfgw.getWindEstimate()#.plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "gw-gmrf_" + str(i) + "_wind")
		gmrfgw.getGasUncertainty().plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "gw-gmrf_" + str(i) + "_sigma")
		path_length, rmse, nlml = compareWithGroundtruth(gmrfgw, environment.gas, mask, partial_observations)
		SCENARIO += [scenario]
		GDM += ["GW-GMRF"]
		PATH += [path_length]
		METRIC += ["RMSE"]
		VALUE += [rmse]
		SCENARIO += [scenario]
		GDM += ["GW-GMRF"]
		PATH += [path_length]
		METRIC += ["NLML"]
		VALUE += [nlml]

	data = pandas.DataFrame({'scenario': SCENARIO,
							 'GDM': GDM,
							 'path': PATH,
							 'metric': METRIC,
							 'value': VALUE})

	data.to_csv('/home/andy/tmp/quick_'+scenario+'.csv')


def plot(scenario):
	data = pandas.read_csv('/home/andy/tmp/quick_'+scenario+'.csv')
	size=(4.4,2.75)
	sns.set_style("whitegrid", {"font.family": "Times New Roman"})
	font_size = 12
	plt.rcParams["xtick.labelsize"] = font_size
	plt.rcParams["ytick.labelsize"] = font_size
	plt.rcParams["legend.fontsize"] = font_size
	plt.rcParams['figure.dpi'] = 300
	#plt.rcParams["font.family"] = "serif"
	fig, ax = plt.subplots(figsize=size)

	print("RMSE")
	plt.ylim(0.0, 0.3)
	#plt.yticks([0,0.1,0.2])
	sns.lineplot(x='path', y='value', hue='GDM', ci=None, data=data[data.metric=="RMSE"], palette=sns.color_palette(["#00a933", "#ff420e", "#2a6099"]))
	#sns.despine()
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:])
	plt.savefig("/home/andy/tmp/"+scenario+"_rmse.svg", format="svg")

	print("NLML")
	fig, ax = plt.subplots(figsize=size)
	data = data[data['GDM'] != "KDM+V/W"]
	plt.ylim(0, 40000)
	ax = sns.lineplot(x='path', y='value', hue='GDM', ci=None, data=data[data.metric=="NLML"], palette=sns.color_palette(["#00a933", "#2a6099"]))
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:])
	plt.savefig("/home/andy/tmp/"+scenario+"_nlml.svg", format="svg")


"""


partial_observations = observations[0:i]
gmrfg  = GMRF_Gas_Efficient(ccm)
gmrfg.addObservation(observations)
gmrfg.estimate()
gmrfg.getGasEstimate().plot(mask=1, vmax=max_gas, save="/home/andy/tmp/gmrf/" + "g-gmrf_all_mean")
gmrfg.getGasUncertainty().plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "g-gmrf_all_sigma")
compareWithGroundtruth(gmrfg, environment.gas, mask)

kdmvw = KDM_VW(ccm)
kdmvw.addObservation(observations)
kdmvw.estimate()
kdmvw.getGasEstimate().plot(mask=1, vmax=max_gas, save="/home/andy/tmp/gmrf/" + "kdm_all_mean")
kdmvw.getGasUncertainty().plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "kdm_all_sigma")
compareWithGroundtruth(kdmvw, environment.gas, mask)


gmrfgw = GMRF_Gas_Wind_Efficient(ccm)
gmrfgw.addObservation(observations)
gmrfgw.estimate()
gmrfgw.getGasEstimate().plot(mask=1, vmax=max_gas, save="/home/andy/tmp/gmrf/" + "gw-gmrf_all_mean")
gmrfgw.getWindEstimate().plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "gw-gmrf_all_wind")
gmrfgw.getGasUncertainty().plot(mask=1, vmax=1, save="/home/andy/tmp/gmrf/" + "gw-gmrf_all_sigma")
compareWithGroundtruth(gmrfgw, environment.gas, mask)
"""

#test()
plot("corridor_quick")
plot("corridor_full")
plot("lab4_full")
print("DONE!")