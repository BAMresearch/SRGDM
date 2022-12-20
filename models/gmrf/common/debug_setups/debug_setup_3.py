from common.environments.corridor_1_small import corridor_1_small
from common import Observation


obstacle_map = corridor_1_small.obstacles

positions = [(2.5, 0.5),
             (2.5, 2.5),
             (2.5, 3.5),
             (0.5, 4.0),
             (4.0, 4.0)]

observations = corridor_1_small.getObservation(positions)

