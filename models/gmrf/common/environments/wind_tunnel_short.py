from gdm.common import ObstacleMap, CellConnectivityMap
import os
import numpy as np


script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/wind_tunnel_short/wind_tunnel.pgm"
pgm_file = script_path + "/" + relative_pgm_file
print("Loading wind tunnel... ", end =" ")
assert os.path.isfile(pgm_file)
wind_tunnel_short = ObstacleMap.fromPGM(pgm_file, resolution=0.1)
assert wind_tunnel_short.size == (12,6)
print("Done")



script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/wind_tunnel_short/wind_tunnel_100.pgm"
pgm_file = script_path + "/" + relative_pgm_file
print("Loading wind tunnel... ", end =" ")
assert os.path.isfile(pgm_file)
wind_tunnel_short_100 = ObstacleMap.fromPGM(pgm_file, resolution=1)
assert wind_tunnel_short_100.size == (12,6)
print("Done")



script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/wind_tunnel_short/wind_tunnel_25.pgm"
pgm_file = script_path + "/" + relative_pgm_file
print("Loading wind tunnel... ", end =" ")
assert os.path.isfile(pgm_file)
wind_tunnel_short_025 = ObstacleMap.fromPGM(pgm_file, resolution=0.25)
assert wind_tunnel_short_025.size == (12,6)
print("Done")



if __name__ == "__main__":
	wind_tunnel_short.plot()
	wind_tunnel_short_100.plot()
	wind_tunnel_short_025.plot()



