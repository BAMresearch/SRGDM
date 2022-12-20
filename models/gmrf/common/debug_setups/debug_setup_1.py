import numpy as np
from gdm.common import Observation, ObstacleMap


m4 = np.array(((0,0),
               (0,0)))

m8 = np.array(((0,0,0,0),
               (0,0,0,0),
               (0,0,0,0),
               (0,0,0,0)))

m16 = np.array(((0,0,0,0,0,0,0,0),
                (0,0,0,0,0,0,0,0),
                (0,0,0,0,0,0,0,0),
                (0,0,0,0,0,0,0,0),
                (0,0,0,0,0,0,0,0),
                (0,0,0,0,0,0,0,0),
                (0,0,0,0,0,0,0,0),
                (0,0,0,0,0,0,0,0)))

om_4 = ObstacleMap.fromMatrix(m4, resolution=0.5)
om_8 = ObstacleMap.fromMatrix(m8, resolution=0.25)
om_16 = ObstacleMap.fromMatrix(m16, resolution=0.125)

observations =[Observation((0.275,0.275), gas=0.0, wind=(-1, 1), data_type='gas+wind'),
	           Observation((0.725,0.725), gas=1.0, wind=(-1, 1), data_type='gas+wind')]
