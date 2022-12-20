from gdm.common import environment
import os
import numpy as np


## CORRIDOR 1 ------------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_csv_file = "../data/test_environments/corridor/1/data.csv"
relative_pgm_file = "../data/test_environments/corridor/1/obstacles.pgm"
csv_file = script_path + "/" + relative_csv_file
pgm_file = script_path + "/" + relative_pgm_file


print("Loading corridor_1... ", end =" ")
assert os.path.isfile(csv_file)
assert os.path.isfile(pgm_file)
corridor_1 = environment.EnvironmentGroundTruth.fromGadenCsv(csv_file)
corridor_1.obstacles.loadPGM(pgm_file)
corridor_1.gas.normalize()
assert corridor_1.size == (15,5)
assert corridor_1._num_cells == 7500
print("Done")



## CORRIDOR 1 0.25m ------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/corridor/1/obstacles_025.pgm"
pgm_file = script_path + "/" + relative_pgm_file
corridor_1_obstacles_025 = environment.ObstacleMap.fromPGM(pgm_file, resolution=0.25)




## CORRIDOR 1 0.5m ------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/corridor/1/obstacles_050.pgm"
pgm_file = script_path + "/" + relative_pgm_file
corridor_1_obstacles_050 = environment.ObstacleMap.fromPGM(pgm_file, resolution=0.5)



if __name__ == "__main__":
	corridor_1.plot()


