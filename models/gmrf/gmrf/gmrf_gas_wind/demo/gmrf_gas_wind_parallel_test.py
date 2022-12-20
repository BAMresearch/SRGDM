from gdm.gmrf import gmrf_gas_wind
from gdm.common.debug_setups.debug_setup_2 import obstacle_map, observations


if(__name__ is "__main__"):

	ggw = gmrf_gas_wind.GMRF_Gas_Wind_Parallel(obstacle_map)
	ggw.addObservation(observations)
	ggw.estimate()
	ggw.gas.plot()
	ggw.wind.plot()
	ggw.computeUncertainty()
	ggw.gas_uncertainty.plot()
	ggw.toNormalDistributionMap().plot()