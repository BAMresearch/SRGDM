from models.kernel_dmv.td_kernel_dmvw import TDKernelDMVW
import torch
import numpy as np


class KernelDMV:
    def __init__(self):
        min_x = 0  
        min_y = 0
        max_x = 30#np.max(notnan[:,0])
        max_y = 25#np.max(notnan[:,1])

        cell_size = 1
        wind_scale = 0
        time_scale = 0

        # Create a 2D mask (like for the neural networks)
        mask = torch.zeros([30,25])
        n = 5
        for row in range(int(n/2), mask.shape[0], n): 
            for col in range(int(n/2), mask.shape[1], n):
                mask[row, col] = 1

        # Get the positions as a 1D array
        positions = torch.tensor(range(30*25))*mask.flatten()
        self.positions = positions.nonzero().squeeze().numpy()

        self.wind_speeds = torch.zeros(len(positions))
        self.wind_directions = torch.zeros(len(positions))
        self.timestamps = range(len(positions))

        kernel_size = 2.5*cell_size
        evaluation_radius = 5*kernel_size

        kernel = TDKernelDMVW(min_x, min_y, 
                              max_x, max_y, 
                              cell_size,
                              kernel_size,
                              wind_scale,
                              time_scale,
                              confidence_scale=1,
                              real_time=False,
                              low_confidence_calculation_zero=True,
                              evaluation_radius=evaluation_radius)
        self.kernel = kernel

    def calculate_old(self, y):
        """ Calculate the KDM based on specified positions and return mean map. """
        y = y.numpy()
        notnan = np.argwhere(np.isnan(y)==False)
        obss= y[notnan[:,0],notnan[:,1]] 
        self.kernel.set_measurements(
            notnan[self.positions, 0], notnan[self.positions, 1], 
            obss[self.positions], 
            self.timestamps, self.wind_speeds, self.wind_directions)
        self.kernel.calculate_maps()
        
        return torch.tensor(self.kernel.mean_map[0:30,0:25])

    def calculate(self, X):
        """ Calculate the KDM based on specified positions and return mean map. """
        X = self.sparsify_img(X) 
        X = X.numpy()      
        
        notnan = np.argwhere(np.isnan(X)==False)
        obss= X[notnan[:,0],notnan[:,1]]        
        self.kernel.set_measurements(
            notnan[self.positions, 0], notnan[self.positions, 1], 
            obss[self.positions], 
            self.timestamps, self.wind_speeds, self.wind_directions)
        self.kernel.calculate_maps()
        
        return torch.tensor(self.kernel.mean_map[0:30,0:25])    
    
    def sparsify_img(self, X):
        X = X.squeeze()
        sparse_x = torch.zeros([30,25])
        n = 5
        for row, i in zip(range(int(n/2), sparse_x.shape[0], n), range(6)): 
            for col, j in zip(range(int(n/2), sparse_x.shape[1], n), range(5)):
                sparse_x[row, col] = X[i,j]
        return sparse_x