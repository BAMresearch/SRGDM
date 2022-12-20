import sys

sys.path.insert(0, "../../../..")
from gdm.gmrf import GMRF_Gas_Wind, GMRF_Gas_Wind_Efficient
from gdm.common.environments.small_office_3 import small_office_3
from gdm.utils.benchmarking import ExecutionTimer
import random
import cProfile


def test():
    small_office_3.gas.normalize()

    observations = []
    for i in range(0, 100):
        position = (random.uniform(0, 10), random.uniform(0, 10))
        observations += [small_office_3.getObservation(position)]

    ggw = GMRF_Gas_Wind_Efficient(small_office_3.obstacles, resolution=0.1)
    ggw.addObservation(observations)
    ggw.addObservation(small_office_3.getObservation((2.5,3)))
    ggw.addObservation(small_office_3.getObservation((5,3)))
    ggw.addObservation(small_office_3.getObservation((8, 3)))
    ggw.addObservation(small_office_3.getObservation((5,8)))

    t0 = ExecutionTimer("Gas-Wind TOTAL")
    t1 = ExecutionTimer("Gas-Wind estimation")
    ggw.estimate()
    t1.getElapsed()

    t2 = ExecutionTimer("Gas-Wind Uncertainty")
    ggw.computeUncertainty()
    t2.getElapsed()
    t0.getElapsed()
    print("")

    ggw.getGasEstimate().plot(vmax=0.25)
    #ggw.getWindEstimate().plot(vmax=1, interpol=1)
    uncertainty = ggw.getGasUncertainty()
    uncertainty.plot()
    uncertainty._data = np.sqrt(uncertainty._data)
    uncertainty.plot()


# ggw.toNormalDistributionMap().plot()

# small_office_3.gas.plot(vmax=1)


if (__name__ == "__main__"):
   test()
