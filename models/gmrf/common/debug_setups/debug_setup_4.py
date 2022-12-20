from gdm.common import ObstacleMap
import os
from gdm.common import Observation

## Load obstacle map
script_path = os.path.dirname(os.path.realpath(__file__))
relative_pgm_file = "../data/test_environments/fork/obstacles.pgm"
pgm_file = script_path + "/" + relative_pgm_file
obstacle_map = ObstacleMap.fromPGM(pgm_file, resolution=0.1)


## Create observations
observations = [Observation((2.5, 0.1), gas=1, wind=( 0,1), data_type='gas+wind'),
                Observation((1, 0.5), gas=0, wind=( 0,0), data_type='gas+wind'),
                Observation((4, 0.5), gas=0, wind=( 0,0), data_type='gas+wind')]


if __name__ == "__main__":
	obstacle_map.plot()
