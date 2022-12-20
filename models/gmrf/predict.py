import torch
import sys

# This is ugly. Should be done without sys.path.append
#sys.path.append("/home/nicolas/git/gdm")
#sys.path.append("models/gmrf")

#import os
# Get the current relative path
#current_path = os.path.dirname(os.path.abspath(__file__))

# Append the current relative path to the system path
#sys.path.append(current_path)

#from gmrf import GMRF_Gas_Efficient
#from common.obstacle_map import ObstacleMap
#from common.observation import Observation

from models.gmrf.common.obstacle_map import ObstacleMap
from models.gmrf.common.observation import Observation
from models.gmrf.gmrf.gmrf_gas.gmrf_gas import GMRF_Gas_Efficient

#om = ObstacleMap(dimensions=2, size=(25,30), resolution=1) # axes are swapped

#def calculate_GMRF(y):
#    """ Calculates the distribution with GMRF. Takes true distribution as input, but grabs the positions of the sparse sensor network.
#    Must be adapted, if different sampling positions are desired."""
#    
#    g = GMRF_Gas_Efficient(om, resolution=1)
#    n = 5
##    for row in range(int(n/2), 30, n): 
#        for col in range(int(n/2), 25, n):
#            conc = y[row][col]
#            obs = Observation(position=((col,30-row)), gas=conc) # adjust the y axis
#            g.addObservation(obs)
#    g.estimate()
#    return torch.tensor(g.getGasEstimate()._data)

import copy

class myGMRF():
    def __init__(self):
        om = ObstacleMap(dimensions=2, size=(25,30), resolution=1) 
        self.g = GMRF_Gas_Efficient(om, resolution=1)
        
    def calculate(y):
        """ Calculates the distribution with GMRF. Takes true distribution as input, but grabs the positions of the sparse sensor network.
        Must be adapted, if different sampling positions are desired."""
        g_c = copy.deepcopy(self.g)
        
        for row in range(int(n/2), 30, n): 
        for col in range(int(n/2), 25, n):
            conc = y[row][col]
            obs = Observation(position=((col,30-row)), gas=conc) # adjust the y axis
            g_c.addObservation(obs)
        g_c.estimate()
    return torch.tensor(g_c.getGasEstimate()._data)