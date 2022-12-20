from gdm.common import ObstacleMap, CellConnectivityMap
import os
import numpy as np


script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/wind_tunnel_obstacles_1/wind_tunnel.pgm"
pgm_file = script_path + "/" + relative_pgm_file
print("Loading wind tunnel... ", end =" ")
assert os.path.isfile(pgm_file)
wind_tunnel_obstacles_1 = ObstacleMap.fromPGM(pgm_file, resolution=0.1)
assert wind_tunnel_obstacles_1.size == (12,6)
print("Done")



script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/wind_tunnel_obstacles_1/wind_tunnel_25.pgm"
pgm_file = script_path + "/" + relative_pgm_file
print("Loading wind tunnel... ", end =" ")
assert os.path.isfile(pgm_file)
wind_tunnel_obstacles_1_025 = ObstacleMap.fromPGM(pgm_file, resolution=0.25)
assert wind_tunnel_obstacles_1_025.size == (12,6)
print("Done")



script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/wind_tunnel_obstacles_1/wind_tunnel_coarse.pgm"
pgm_file = script_path + "/" + relative_pgm_file
print("Loading wind tunnel... ", end =" ")
assert os.path.isfile(pgm_file)
wind_tunnel_obstacles_1_coarse = ObstacleMap.fromPGM(pgm_file, resolution=0.1)
assert wind_tunnel_obstacles_1_coarse.size == (12,6)
print("Done")



if __name__ == "__main__":
	wind_tunnel_obstacles_1.plot()
	wind_tunnel_obstacles_1_025.plot()
	wind_tunnel_obstacles_1_coarse.plot()



