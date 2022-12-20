import natsort
import os
import numpy as np
import torch
from torch.utils import data


class RealGasDataSet(data.Dataset):   
    """
    Dataset that returns a sequence of real gas distribution data, obtained during experiments.
    Output:
        X - Gas distribution data of length `seq_len` of time step T and the preceeding time steps.
    """
    def __init__(self, root, seq_len=1, nth_frame=1, sliding_step=1):
        self.root = root
        self.seq_len = seq_len        # how many time steps shall be predicted
        
        # Load the data. The data tensor is structured: [n_sources, n_images, height, width]
        # n_sources ->  individual simulations of sources
        # n_images  ->  images of each simulation run 
        x = torch.load(root)        
        # Select only the images that we care about (ruled by nth frame)
        idxs = torch.tensor(range(x.shape[1])[::nth_frame])
        x = torch.index_select(x, 1, idxs)

        # Get the number of sequences, that each source simulation generates
        n_seq = x[0].unfold(0, seq_len, sliding_step).size()[0]

        # Create an empty tensor that is going to be filled up in the loop
        # Dims: [source simulations, sequences, images_per_sequence, width, height]
        new_x = torch.empty([x.size()[0], n_seq, seq_len, 6, 5])
        new_x = new_x.type_as(x)

        # Loop over the source simulations (120 = 30 positions * 4 seasons)
        for i in range(x.shape[0]):
            sim = x[i]
            
            # Slice the continuous sequence into sequences of specific length (seq_len)
            sim = sim.unfold(0, seq_len, sliding_step)
            
            # Change ordering of dimensions
            sim = torch.permute(sim, (0, 3, 1, 2))
            new_x[i] = sim

        self.data = new_x.reshape(-1, seq_len, 6, 5)    
        
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.unsqueeze(1)
    
        X = sample
            
        return X
    
