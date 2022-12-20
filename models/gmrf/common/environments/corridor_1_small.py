from common import environment
import os
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
relative_csv_file = "../data/test_environments/corridor/1_small/data.csv"
relative_pgm_file = "../data/test_environments/corridor/1_small/obstacles.pgm"
csv_file = script_path + "/" + relative_csv_file
pgm_file = script_path + "/" + relative_pgm_file


print("Loading corridor_1_small... ", end =" ")
assert os.path.isfile(csv_file)
assert os.path.isfile(pgm_file)
corridor_1_small = environment.EnvironmentGroundTruth.fromGadenCsv(csv_file)
corridor_1_small.obstacles.loadPGM(pgm_file)
corridor_1_small.gas.normalize()
print("Done")




from .. import CellConnectivityMap
low_res_conn = np.ones((25,25))
low_res_conn[1, 2]   = low_res_conn[2, 1]   = 0
low_res_conn[6, 7]   = low_res_conn[7, 6]   = 0
low_res_conn[16, 17] = low_res_conn[17, 16] = 0
low_res_conn[21, 22] = low_res_conn[22, 21] = 0
corridor_1_small_ccm_100 = CellConnectivityMap.fromMatrix(low_res_conn, size=(5,5), resolution=1)



"""
from gdm.common import CellConnectivityMap
low_res_conn = np.ones((100,100))
low_res_conn[3, 4]   = low_res_conn[4, 3]   = 0
low_res_conn[13,14]   = low_res_conn[14,13]   = 0
low_res_conn[23,24]   = low_res_conn[24,23]   = 0
low_res_conn[33,24]   = low_res_conn[34,33]   = 0
low_res_conn[63,64]   = low_res_conn[64,63]   = 0
low_res_conn[73,74]   = low_res_conn[74,73]   = 0
low_res_conn[83,84]   = low_res_conn[84,83]   = 0
low_res_conn[93,94]   = low_res_conn[94,93]   = 0

corridor_1_small_coarse = CellConnectivityMap.fromMatrix(low_res_conn, size=(5,5), resolution=0.5)
"""



if __name__ == "__main__":
	corridor_1_small.plot()
