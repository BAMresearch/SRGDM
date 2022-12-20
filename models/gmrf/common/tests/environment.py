from gdm.gdm.common import environment
"""
o = environment.ObstacleMap.fromPGM("../data/test_environments/small_office/3/obstacles.pgm", 0.1).normalize().invertValues()
o.plot()
print(type(o))
print(o.toMatrix())
cell_1 =(1,1)
cell_2 = (2,2)
print(o.hasObstacleBetweenCells(cell_1, cell_2))



e = environment.EnvironmentGroundTruth.fromGadenCsv("../data/test_environments/small_office/3/data.csv")
e.gas.plot()
e.wind.plot()
e.obstacles.plot()
print(e.getCell((10,10)).gas)
print(e.getObservation((5,5)).gas)

"""
"""
from gdm.common.debug_setups.debug_setup_3 import obstacle_map, observations
d = obstacle_map.toDistanceMap((1,4))#.plot(vmax=5)
path = d.getShortestPathToPosition((1,2))
d.setPosition(path, 5).plot(vmax=10)



from ....gdm.common.debug_setups.debug_setup_0 import obstacle_map, observations
ocm = environment.ObstacleConnectivityMap(obstacle_map.size, obstacle_map.resolution)
ocm.loadObstacleMap(obstacle_map)
print(ocm._conectivity)

"""