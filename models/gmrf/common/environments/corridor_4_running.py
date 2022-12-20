import os
import pickle

from gdm.common import environment

script_path = os.path.dirname(os.path.realpath(__file__))
relative_serialized = "../data/test_environments/corridor/4/serialized.pickle"
serialized_file = script_path + "/" + relative_serialized

print("Loading corridor_4...")
if not os.path.isfile(serialized_file):
    print("from corridor 1 and 2... ", end="")
    from corridor_1_running import corridor_1
    from corridor_2_running import corridor_2

    assert len(corridor_1.environments) == len(corridor_2.environments)

    corridor_4 = environment.EnvironmentRunningGroundTruth(dimensions=corridor_1.dimensions,
                                                           size=corridor_1.size,
                                                           resolution=corridor_1.resolution,
                                                           period=corridor_1.period)

    corridor_4.environments = []
    num_snapshots = len(corridor_1.environments)
    for i in range(num_snapshots):
        me1_g = corridor_1.environments[i].gas.toMatrix()
        me2_g = corridor_2.environments[i].gas.toMatrix()
        me4_g = (me1_g + me2_g) / 2

        me1_w_i = corridor_1.environments[i].wind.toMatrix()[0]
        me2_w_i = corridor_2.environments[i].wind.toMatrix()[0]
        me4_w_i = (me1_w_i + me2_w_i) / 2

        me1_w_j = corridor_1.environments[i].wind.toMatrix()[1]
        me2_w_j = corridor_2.environments[i].wind.toMatrix()[1]
        me4_w_j = (me1_w_j + me2_w_j) / 2

        me1_o = corridor_1.environments[i].obstacles.toMatrix()
        me2_o = corridor_2.environments[i].obstacles.toMatrix()
        me4_o = (me1_o + me2_o) / 2

        corridor_4.environments += [environment.EnvironmentGroundTruth(dimensions=corridor_1.dimensions,
                                                                      size=corridor_1.size,
                                                                      resolution=corridor_1.resolution)]

        corridor_4.environments[-1].gas.loadMatrix(me4_g)
        corridor_4.environments[-1].wind.loadMatrix((me4_w_i, me4_w_j))
        corridor_4.environments[-1].obstacles = corridor_1.obstacles

    corridor_4.gas = corridor_4.environments[-1].gas
    corridor_4.wind = corridor_4.environments[-1].wind
    corridor_4.obstacles = corridor_4.environments[-1].obstacles

    assert corridor_4.size == (15, 5)
    assert corridor_4._num_cells == 7500
    pickle_out = open(serialized_file, "wb")
    pickle.dump(corridor_4, pickle_out)
    pickle_out.close()
else:
    print("from pickle file... ", end="")
    pickle_in = open(serialized_file, "rb")
    corridor_4 = pickle.load(pickle_in)
print("Done")

if __name__ == "__main__":
    corridor_4.plot()
    print(len(corridor_4.environments))
    corridor_4.environments[0].plot()
