import sys

sys.path.append("/home/nicolas/git/gdm/")
from gmrf import GMRF_Gas, GMRF_Gas_Efficient
#from gdm.common.environments.corridor_1 import corridor_1
from common.environments.small_office_3 import small_office_3
from utils.benchmarking import ExecutionTimer
import random


def test():
    environment = small_office_3
    environment.gas.normalize()

    observations = []
    for i in range(0, 100):
        max_x = environment.size[0]
        max_y = environment.size[1]
        position = (random.uniform(0, max_x), random.uniform(0, max_y))
        observations += [environment.getObservation(position)]

    t = ExecutionTimer("[GMRF-Gas] Initialization")
    g = GMRF_Gas_Efficient(environment.obstacles, resolution=0.1)
    #g.addObservation(observations)
    obs = environment.getObservation((5, 1))
    obs.gas=1
    g.addObservation(obs)
    t.getElapsed()

    t = ExecutionTimer("[GMRF-Gas] Gas estimation")
    g.estimate()
    t.getElapsed()

    t = ExecutionTimer("[GMRF-Gas] Gas uncertainty")
    g.computeUncertainty()
    t.getElapsed()

    g.getGasEstimate().plot()
    g.getGasUncertainty().plot(vmax=1)


if (__name__ == "__main__"):
    test()
