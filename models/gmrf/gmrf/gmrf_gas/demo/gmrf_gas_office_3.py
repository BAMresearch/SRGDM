if __name__ == "__main__":
    import sys

    sys.path.insert(0, "../../../..")
    from gdm.gmrf import GMRF_Gas
    from gdm.common.environments.small_office_3 import small_office_3
    from gdm.utils.benchmarking import ExecutionTimer
    import random
    from gdm.utils.report import getInstanceSize

    small_office_3.gas.normalize()
    # small_office_3.gas.loadMatrix(small_office_3.gas.toMatrix()*100)

    observations = []
    for i in range(0, 1000):
        position = (random.uniform(0, 10), random.uniform(0, 10))
        observations += [small_office_3.getObservation(position)]

    t = ExecutionTimer("[GMRF-Gas] Initialization")
    g = GMRF_Gas(small_office_3.obstacles)
    g.addObservation(observations)
    t.getElapsed()

    t = ExecutionTimer("[GMRF-Gas] Gas estimation")
    g.estimate()
    t.getElapsed()

    t = ExecutionTimer("[GMRF-Gas] Gas uncertainty")
    g.computeUncertainty()
    t.getElapsed()

    g.toNormalDistributionMap().plot()
