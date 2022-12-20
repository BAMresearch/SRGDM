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
from gdm.common import ObstacleMap, CellConnectivityMap


GAS_MAX = 1


###############################################################################
def filterArray(array, window_width=3):
	filtered = array.copy()
	k = int(window_width / 2)

	if (k > 0):
		for i in range(k, array.shape[0] - k):
			filtered[i] = array[i - k:i + k + 1].sum() / (2 * k + 1)

	return filtered

def filterGasSensor(gas):
	## CONFIGURATION PARAMETERS
	a_rise = 0
	a_fall = 0
	b_rise = 2
	b_fall = 24
	# n = 1 #### NOT IMPLEMENTE YET!!!
	t = 0.5

	## ADJUST CONFIGURATION PARAMETERS
	a_rise /= t
	a_fall /= t
	b_rise /= t
	b_fall /= t

	## PROCESS DATA
	gas = filterArray(gas, 5)
	max_gas = gas.max()
	min_gas = gas.min()
	gradient = np.gradient(gas)
	gradient = filterArray(gradient, 5)
	T = np.zeros(gas.shape[0])

	## COMPUTE COMPENSATION PARAMETER DEPENDING ON POSITIVE OR NEGATIVE GRADIENT
	T[gradient > 0] = (a_rise * (max_gas - gas[gradient > 0]) + b_rise) / t
	T[gradient <= 0] = (a_fall * (max_gas - gas[gradient <= 0]) + b_fall) / t;

	## COMPUTE GAS PREDICTION AND LIMIT TO REAL MAX AND MIN
	prediction = gas + T * gradient
	prediction = filterArray(prediction, 5)
	prediction[prediction > max_gas] = max_gas
	prediction[prediction < min_gas] = min_gas

	## PLOT FILTERED DATA
	# plt.plot(prediction)
	# plt.plot(gas)
	# plt.plot(gradient)

	return prediction


def rotateVector(vector,angle):
	x = vector[0]*np.cos(angle) - vector[1]*np.sin(angle)
	y = vector[0]*np.sin(angle) + vector[1]*np.cos(angle)
	return (x,y)



###############################################################################


def getObservations(file, position_offset=(0,0), rotate=0, interval=(0,0), strech=(1,1)):

	data = pandas.read_csv(file, sep=',', header=0)
	gas = np.array(data["#gas"])
	filtered_gas = filterGasSensor(gas)
	data["#gas"] = filtered_gas

	if not interval[0] == interval[1] == 0:
		data = data[interval[0]:interval[1]]



	observations = []
	for index, row in data.iterrows():
		y      = row['#odom_x']*strech[1]
		x      = row['#odom_y']*strech[0]
		w      = row['#odom_w']
		gas    = row['#gas']
		wind_s = row['#wind_s']
		wind_d = row['#wind_d']-np.pi

		wind_x = wind_s*np.sin(wind_d)
		wind_y = wind_s*np.cos(wind_d)-0.95*row["#speed"]
		wind = rotateVector((wind_x, wind_y), w-rotate)

		position = (x, y)
		position = rotateVector(position, rotate)
		position = (position[0] + position_offset[0], position[1] + position_offset[1])

		obs = Observation(position=position, gas=gas, wind=wind, data_type="gas+wind")
		observations += [obs]

	return observations



def plotReal(csv_file, map_file, resolution, position_offset, interval=(0,0), name="", strech=(1,1), rotate=0):
	observations = getObservations(csv_file, position_offset=position_offset, rotate=rotate, interval=interval,strech=strech)
	ccm = ObstacleMap.fromPGM(map_file, resolution=resolution).toCellConnectivityMap()


	gmrfg = GMRF_Gas_Efficient(ccm)
	gmrfg.addObservation(observations)
	gmrfg.getGasEstimate().plot(vmax=GAS_MAX, mask=1, save='/home/andy/tmp/' + name + '_ggmrf_mean.png')
	gmrfg.getGasUncertainty().plot(vmax=1, mask=1, save='/home/andy/tmp/' + name + '_ggmrf_sigma.png')

	kdmvw = KDM_VW(ccm)
	kdmvw.addObservation(observations)
	kdmvw.getGasEstimate().plot(vmax=GAS_MAX, mask=1, save='/home/andy/tmp/' + name + '_kdm_mean.png')
	kdmvw.getGasUncertainty().plot(vmax=1, mask=1, save='/home/andy/tmp/' + name + '_kdm_sigma.png')

	gmrfgw  = GMRF_Gas_Wind_Efficient(ccm)
	gmrfgw.addObservation(observations)
	gmrfgw.getGasEstimate().plot(vmax=GAS_MAX, mask=1, save='/home/andy/tmp/' + name + '_gwgmrf_mean.png')
	gmrfgw.getWindEstimate().plot(vmax=1, mask=1, save='/home/andy/tmp/' + name + '_gwgmrf_wind.png')
	#gmrfgw.getGasUncertainty().plot(vmax=1, mask=1, save='/home/andy/tmp/' + name + '_gwgmrf_sigma.png')


	"""
	gmrfg = GMRF_Gas_Efficient(ccm)
	gmrfg.addObservation(observations)
	gmrfg.getGasEstimate().plot(vmax=GAS_MAX, mask=1)
	gmrfg.getGasUncertainty().plot(vmax=1, mask=1)
	"""





#observations = getObservations('/home/andy/tmp/csv/corridor_real.csv', position_offset=(10, 5), rotate=0, interval=(2500,3200))
#observations = getObservations('/home/andy/tmp/csv/test_4.csv', position_offset=(7.5, 4.5))

#ccm = ObstacleMap.fromPGM("/home/andy/tmp/csv/mapa_mapir2.pgm", resolution=0.1).toCellConnectivityMap()
#matrix = np.zeros((100,100))
#ccm = ObstacleMap.fromMatrix(matrix, resolution=0.2).toCellConnectivityMap()

#plotReal(name="mapir_0",    csv_file='/home/andy/tmp/csv/test_4.csv',        map_file="/home/andy/tmp/csv/mapa_mapir2.pgm",    resolution=0.1, position_offset=(7.5, 4.8), strech=(1,1), interval=(400, 500))
#plotReal(name="mapir_1",    csv_file='/home/andy/tmp/csv/test_4.csv',        map_file="/home/andy/tmp/csv/mapa_mapir2.pgm",    resolution=0.1, position_offset=(7.5, 4.8), strech=(1,1), interval=(400, 900))
#plotReal(name="corridor", csv_file='/home/andy/tmp/csv/corridor_real.csv', map_file="/home/andy/tmp/csv/corridor_real2.pgm", resolution=0.1, position_offset=(11.4, -0.1), interval=(615, 980))#interval=(2518, 3255))
##plotReal(name="corridor", csv_file='/home/andy/tmp/csv/corridor_real.csv', map_file="/home/andy/tmp/csv/corridor_real2.pgm", resolution=0.1, position_offset=(11.3, 0), interval=(620,950))
#plotReal(name="corridor_0", csv_file='/home/andy/tmp/csv/corridor_real.csv', map_file="/home/andy/tmp/csv/corridor_real2.pgm", resolution=0.1, position_offset=(11.4, -0.1), interval=(615, 680))#interval=(2518, 3255))


#plotReal(name="banio4", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(0,0), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))
#plotReal(name="banio4b1", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(340,390), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))
#plotReal(name="banio4b2", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(340,445), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))
plotReal(name="banio4b3", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(340,470), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))
#plotReal(name="banio4b4", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(340,510), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))

#plotReal(name="banio4c2", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(500,1250), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))
#plotReal(name="banio4c2", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(900,1250), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))

#plotReal(name="banio41", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(0,70), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))
#plotReal(name="banio42", csv_file='/home/andy/tmp/csv/banios/banios_4.csv', interval=(75,150), map_file="/home/andy/tmp/csv/banios/banios2.pgm", resolution=0.1, rotate=np.pi/2, position_offset=(9, 6), strech=(1,-1))
