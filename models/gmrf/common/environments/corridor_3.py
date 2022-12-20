from gdm.common import environment
import os
import numpy as np


## CORRIDOR 3 ------------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_csv_file = "../data/test_environments/corridor/3/data.csv"
relative_pgm_file = "../data/test_environments/corridor/3/obstacles.pgm"
csv_file = script_path + "/" + relative_csv_file
pgm_file = script_path + "/" + relative_pgm_file


print("Loading corridor_3... ", end =" ")
assert os.path.isfile(csv_file)
assert os.path.isfile(pgm_file)
corridor_3 = environment.EnvironmentGroundTruth.fromGadenCsv(csv_file)
corridor_3.obstacles.loadPGM(pgm_file)
assert corridor_3.size == (15,5)
print("Done")



## CORRIDOR 3 0.25m ------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/corridor/3/obstacles_025.pgm"
pgm_file = script_path + "/" + relative_pgm_file

print("Loading corridor_3 obstacle map at 0.25m ... ", end =" ")
corridor_3_obstacles_025 = environment.ObstacleMap.fromPGM(pgm_file, resolution=0.25)
assert corridor_3.size == (15,5)
print("Done")



## CORRIDOR 3 0.5m ------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/corridor/3/obstacles_050.pgm"
pgm_file = script_path + "/" + relative_pgm_file
corridor_3_obstacles_050 = environment.ObstacleMap.fromPGM(pgm_file, resolution=0.5)



## CORRIDOR 3 COARSE (1M) CCM --------------------------------------------------
from gdm.common.environment import CellConnectivityMap
low_res_conn = np.ones((75,75))
low_res_conn[1, 2]   = low_res_conn[2, 1]   = 0
low_res_conn[6, 7]   = low_res_conn[7, 6]   = 0
low_res_conn[16, 17] = low_res_conn[17, 16] = 0
low_res_conn[21, 22] = low_res_conn[22, 21] = 0

low_res_conn[26, 27] = low_res_conn[27, 26] = 0
low_res_conn[31, 32] = low_res_conn[32, 31] = 0
low_res_conn[36, 37] = low_res_conn[37, 36] = 0
low_res_conn[41, 42] = low_res_conn[42, 41] = 0

low_res_conn[56, 57] = low_res_conn[57, 56] = 0
low_res_conn[61, 62] = low_res_conn[62, 61] = 0
low_res_conn[66, 67] = low_res_conn[67, 66] = 0
low_res_conn[71, 72] = low_res_conn[72, 71] = 0

low_res_conn[22, 27] = low_res_conn[27, 22] = 0
low_res_conn[23, 28] = low_res_conn[28, 23] = 0
low_res_conn[24, 29] = low_res_conn[29, 24] = 0

low_res_conn[52, 47] = low_res_conn[47, 52] = 0
low_res_conn[53, 48] = low_res_conn[48, 53] = 0
low_res_conn[54, 49] = low_res_conn[49, 54] = 0

corridor_3_ccm_100 = CellConnectivityMap.fromMatrix(low_res_conn, size=(15,5), resolution=1)





if __name__ == "__main__":
	corridor_3.plot()


