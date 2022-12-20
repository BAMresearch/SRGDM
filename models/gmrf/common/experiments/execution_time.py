from gdm.common.environments.corridor_1_running import corridor_1
from gdm.common.environments.mapir_4b_running import mapir_4b
from gdm.gmrf import GMRF_Gas
from gdm.kdm import KDM_VW
from gdm.gmrf import GMRF_Gas_Wind_Efficient, GMRF_Gas_Efficient
from gdm.common import Lattice2DScalar, NormalDistributionMapper
import gdm.utils.metrics
import sys
import numpy as np
from gdm.common import ObstacleMap, Observation
from gdm.utils.benchmarking import ExecutionTimer
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib




def test():

	N = []
	GDM = []
	Tg = []
	Tu = []

	ns = (200, 100, 70, 50, 31, 20, 10, 3)
	for n in ns:
		print(n)
		matrix = np.zeros((n,n))
		matrix[0,:] = 1
		matrix[n-1,:] = 1
		matrix[:,0] = 1
		matrix[:,n-1] = 1
		obstacles = ObstacleMap.fromMatrix(matrix, resolution=0.1)
		ccm = obstacles.toCellConnectivityMap()

		observations = []
		for i in range(1,n-1):
			observations += [Observation(position=(0.1*i, 0.1*i), gas=i)]

		gmrfg  = GMRF_Gas_Efficient(ccm)
		kdmvw  = KDM_VW(ccm)
		gmrfgw = GMRF_Gas_Wind_Efficient(ccm)

		gmrfg.addObservation(observations)
		kdmvw.addObservation(observations)
		gmrfgw.addObservation(observations)

		t = ExecutionTimer()
		gmrfg.getGasEstimate()
		tg = t.getElapsed()
		t = ExecutionTimer()
		gmrfg.getGasUncertainty()
		tu = t.getElapsed()

		GDM += ["G-GMRF"]
		N += [n]
		Tg += [tg]
		Tu += [tu]

		t = ExecutionTimer()
		kdmvw.getGasEstimate()
		tg = t.getElapsed()
		t = ExecutionTimer()
		kdmvw.getGasUncertainty()
		tu = t.getElapsed()

		GDM += ["KDM-V/W"]
		N += [n]
		Tg += [tg]
		Tu += [tu]

		t = ExecutionTimer()
		gmrfgw.getGasEstimate()
		tg = t.getElapsed()
		t = ExecutionTimer()
		gmrfgw.getGasUncertainty()
		tu = t.getElapsed()

		GDM += ["GW-GMRF"]
		N += [n]
		Tg += [tg]
		Tu += [tu]


	data = pandas.DataFrame({'GDM' : GDM,
							 'N':N,
							 "Time_gas" : Tg,
							 "Time_uncertainty" : Tu})


	data.to_csv('/home/andy/tmp/estimation_time_comparison.csv')



def plot():
	data = pandas.read_csv('/home/andy/tmp/estimation_time_comparison.csv')
	data["Total"] = data['Time_gas'] + data['Time_uncertainty']
	sns.set_style("whitegrid", {"font.family":"Times New Roman"})
	size = (7.5, 3.5)
	font_size = 10
	plt.rcParams["xtick.labelsize"] = font_size
	plt.rcParams["ytick.labelsize"] = font_size
	plt.rcParams["legend.fontsize"] = font_size
	plt.rcParams['figure.dpi'] = 300
	fig, ax = plt.subplots(figsize=size)
	ax.set(yscale="log")
	ax.set_ylim(0.0001, 10)
	data = data[data['N'] != 20]
	data = data[data['N'] != 50]
	data = data[data['N'] != 70]
	data = data[data['N'] != 200]

	sns.barplot(x='N', y='Total', hue='GDM', data=data, linewidth=1, palette=sns.color_palette(["#00a933", "#ff420e", "#2a6099"]))
	plt.savefig("/home/andy/tmp/estimation_time_comparison_total.svg", format="svg")

	sns.barplot(x='N', y='Time_gas', hue='GDM', data=data, linewidth=1, palette=sns.color_palette(["#00a933", "#ff420e", "#2a6099"]))
	plt.savefig("/home/andy/tmp/estimation_time_comparison_gas.svg", format="svg")

	#plt.show()




plot()