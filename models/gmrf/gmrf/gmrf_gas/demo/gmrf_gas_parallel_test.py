import sys
sys.path.append("/home/nicolas/git/gdm/")
from gmrf import gmrf_gas
from common.debug_setups.debug_setup_2 import obstacle_map, observations


if(__name__ is "__main__"):

	g = gmrf_gas.GMRF_Gas_Parallel(obstacle_map)
	g.addObservation(observations)
	g.estimate()
	g.gas.plot()
	g.computeUncertainty()
	g.gas_uncertainty.plot()
