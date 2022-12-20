from gdm.common import environment
import os
import numpy as np
import pickle

script_path = os.path.dirname(os.path.realpath(__file__))
relative_csv_folder = "../data/test_environments/mapir/4b/iterations"
relative_pgm_file = "../data/test_environments/mapir/4b/obstacles.pgm"
relative_serialized = "../data/test_environments/mapir/4b/serialized.pickle"
csv_folder = script_path + "/" + relative_csv_folder
pgm_file = script_path + "/" + relative_pgm_file
serialized_file =  script_path + "/" + relative_serialized


print("Loading mapir_4b... ", end ="")
if not os.path.isfile(serialized_file):
	print("from CSV files... ", end ="")
	assert os.path.isdir(csv_folder)
	assert os.path.isfile(pgm_file)
	mapir_4b = environment.EnvironmentRunningGroundTruth.fromGadenCsvFolder(csv_folder, height=9)
	mapir_4b.obstacles.loadPGM(pgm_file)
	mapir_4b.gas.normalize()
	for e in mapir_4b.environments:
		e.gas.normalize()
	assert mapir_4b.size == (10,11), str(mapir_4b.size)
	assert mapir_4b._num_cells == 11000, str(mapir_4b._num_cells)
	pickle_out = open(serialized_file, "wb")
	pickle.dump(mapir_4b, pickle_out)
	pickle_out.close()
else:
	print("from pickle file... ", end="")
	pickle_in = open(serialized_file, "rb")
	mapir_4b = pickle.load(pickle_in)
print("Done")




if __name__ == "__main__":
	mapir_4b.plot()
	print(len(mapir_4b.environments))
	mapir_4b.environments[0].plot()

	#i=0
	#for env in  mapir_4b.environments:
	#	env.gas.plot(vmax=1, save="/home/andy/tmp/lab4/" + str(i) + "_gt")
	#	i += 1


