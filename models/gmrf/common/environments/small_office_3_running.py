from common import environment
import os
import numpy as np
import pickle


## CORRIDOR 1 ------------------------------------------------------------------
script_path = os.path.dirname(os.path.realpath(__file__))
relative_csv_folder = "../data/test_environments/small_office/3/iterations"
relative_pgm_file = "../data/test_environments/small_office/3/obstacles.pgm"
relative_serialized = "../data/test_environments/small_office/3/serialized.pickle"
csv_folder = script_path + "/" + relative_csv_folder
pgm_file = script_path + "/" + relative_pgm_file
serialized_file =  script_path + "/" + relative_serialized


print("Loading small_office_3... ", end ="")
if not os.path.isfile(serialized_file):
	print("from CSV files... ", end ="")
	assert os.path.isdir(csv_folder)
	assert os.path.isfile(pgm_file)
	small_office_3 = environment.EnvironmentRunningGroundTruth.fromGadenCsvFolder(csv_folder)
	small_office_3.obstacles.loadPGM(pgm_file)
	small_office_3.gas.normalize()
	for e in small_office_3.environments:
		e.gas.normalize()
	assert small_office_3.size == (10.2,10.2)
	assert small_office_3._num_cells == 10404
	pickle_out = open(serialized_file, "wb")
	pickle.dump(small_office_3, pickle_out)
	pickle_out.close()
else:
	print("from pickle file... ", end="")
	pickle_in = open(serialized_file, "rb")
	small_office_3 = pickle.load(pickle_in)
print("Done")



if __name__ == "__main__":
	small_office_3.plot()
	small_office_3.environments[0].plot()


