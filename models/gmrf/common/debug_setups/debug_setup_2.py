from common.environments.small_office_3 import small_office_3
from common import Observation


obstacle_map = small_office_3.obstacles

observations = [Observation((0.2, 9.0),          wind=( 3,0), data_type='wind'),
	            Observation((5.0, 5.0), gas=1.0, wind=(0,-1), data_type='gas+wind'),
				Observation((5.0, 6.0), gas=0.2, wind=(0,-1), data_type='gas+wind'),
                Observation((9.0, 5.0), gas=0.0,              data_type='gas'),
	            Observation((9.0, 9.0), gas=0.0,              data_type='gas'),
	            Observation((8.0, 2.0), gas=1.0,              data_type='gas'),
				Observation((7.7, 2.0), gas=0.2,              data_type='gas'),
	            Observation((0.2, 6.0),          wind=(-3,0), data_type='wind')]
