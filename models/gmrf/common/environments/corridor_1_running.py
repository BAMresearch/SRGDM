import sys
sys.path.append("/home/nicolas/git/gdm/")

from common import environment, ObstacleMap
import os
import numpy as np
import pickle


## CORRIDOR 1 ------------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_csv_folder = "../data/test_environments/corridor/1/iterations"
relative_pgm_file = "../data/test_environments/corridor/1/obstacles.pgm"
relative_serialized = "../data/test_environments/corridor/1/serialized.pickle"
csv_folder = script_path + "/" + relative_csv_folder
pgm_file = script_path + "/" + relative_pgm_file
serialized_file =  script_path + "/" + relative_serialized


print("Loading corridor_1... ", end ="")
if not os.path.isfile(serialized_file):
	print("from CSV files... ", end ="")
	assert os.path.isdir(csv_folder)
	assert os.path.isfile(pgm_file)
	corridor_1 = environment.EnvironmentRunningGroundTruth.fromGadenCsvFolder(csv_folder)
	corridor_1.obstacles.loadPGM(pgm_file)
	corridor_1.gas.normalize()
	for e in corridor_1.environments:
		e.gas.normalize()
	assert corridor_1.size == (15,5)
	assert corridor_1._num_cells == 7500
	pickle_out = open(serialized_file, "wb")
	pickle.dump(corridor_1, pickle_out)
	pickle_out.close()
else:
	print("from pickle file... ", end="")
	pickle_in = open(serialized_file, "rb")
	corridor_1 = pickle.load(pickle_in)

corridor_1_safe_obstacles = ObstacleMap.fromPGM(pgm_file[:-4] + "_big.pgm", resolution=corridor_1.resolution)
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
	#corridor_1.plot()
	#print(len(corridor_1.environments))
	#corridor_1.environments[0].plot()
	#corridor_1_safe_obstacles.plot()
	corridor_1_obstacles_100.plot()

