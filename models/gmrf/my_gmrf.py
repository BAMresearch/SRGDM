import copy
import torch

from models.gmrf.common.obstacle_map import ObstacleMap
from models.gmrf.common.observation import Observation
from models.gmrf.gmrf.gmrf_gas.gmrf_gas import GMRF_Gas_Efficient


class myGMRF():
    def __init__(self):
        om = ObstacleMap(dimensions=2, size=(25,30), resolution=1) 
        self.g = GMRF_Gas_Efficient(om, resolution=1)
        
#    def calculate_old(self, y):
#        """ Calculates the distribution with GMRF. Takes true distribution as input, but grabs the positions of the sparse sensor network.
#        Must be adapted, if different sampling positions are desired."""
#        g_c = copy.deepcopy(self.g)
        
#        n = 5
#        for row in range(int(n/2), 30, n): 
#            for col in range(int(n/2), 25, n):
#                conc = y[row][col]
#                obs = Observation(position=((col,30-row)), gas=conc) # adjust the y axis
#                g_c.addObservation(obs)
#        g_c.estimate()
#        return torch.tensor(g_c.getGasEstimate()._data)
    
    def calculate(self, X):
        """ Calculates the distribution with GMRF. Must be adapted, if different sampling positions are desired."""
        g_c = copy.deepcopy(self.g)
        
        for row in range(6): 
            for col in range(5):
                conc = X[row][col]
                
                x_pos = col*5 + 2
                y_pos = (6-row)*5 - 2
                
                obs = Observation(position=(x_pos,y_pos), gas=conc) # adjust the y axis
                g_c.addObservation(obs)
        g_c.estimate()
        return torch.tensor(g_c.getGasEstimate()._data)