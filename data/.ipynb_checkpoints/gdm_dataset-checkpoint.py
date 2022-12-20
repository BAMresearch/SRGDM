import natsort
import os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import pytorch_lightning as pl


class GasDataSet(data.Dataset):   
    """
    Dataset that returns a sequence of gas distribution data.
    Output:
        X - Masked gas distribution data of length `seq_len` of time step T and the preceeding time steps.
        y - Gas distribution of time step T.
    """
    def __init__(self, root, seq_len=1, nth_frame=2, sliding_step=1):
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
        new_x = torch.empty([x.size()[0], n_seq, seq_len, 30, 25])
        new_x = new_x.type_as(x)

        # Loop over the source simulations (120 = 30 positions * 4 seasons)
        for i in range(x.shape[0]):
            sim = x[i]
            
            # Slice the continuous sequence into sequences of specific length (seq_len)
            sim = sim.unfold(0, seq_len, sliding_step)
            
            # Change ordering of dimensions
            sim = torch.permute(sim, (0, 3, 1, 2))
            new_x[i] = sim

        self.data = new_x.reshape(-1, seq_len, 30, 25)    

        mask = torch.zeros([30,25])
        n = 5
        for row in range(int(n/2), mask.shape[0], n): 
            for col in range(int(n/2), mask.shape[1], n):
                mask[row, col] = 1
        self.mask = mask.repeat(seq_len,1,1,1)
        
        self.transform = transforms.Compose([
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip(),
                        ])
    def downsample(self, X):
        downsampled = torch.zeros(self.seq_len,6,5)
        n = 5
        for img in range(self.seq_len):
            for row_target, row_origin in zip(range(6), range(int(n/2), 30, n)):
                for col_target, col_origin in zip(range(5), range(int(n/2), 25, n)):
                    downsampled[img,row_target, col_target] = X[img][row_origin][col_origin]
        return downsampled
        
    def __len__(self):
        return len(self.data)#-self.seq_len
            
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.unsqueeze(1)
                
        sample = self.transform(sample)

        # Remove channel dimension and add channel dimension before&after downsample
        X = self.downsample(sample.squeeze(1)).unsqueeze(1)
        y = sample[-1]
            
        return X, y
    

class GasDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, seq_len=1, nth_frame=2, sliding_step=1,  num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.nth_frame = nth_frame
        self.batch_size = batch_size
        self.sliding_step = sliding_step
        self.num_workers = num_workers
        self.train_dir = os.path.join(data_dir, 'train.pt')
        self.val_dir = os.path.join(data_dir, 'valid.pt')
        self.test_dir = os.path.join(data_dir, 'test.pt')
        
    def setup(self, stage = None):
        self.train_set = GasDataSet(self.train_dir, 
                                        seq_len=self.seq_len, 
                                        nth_frame=self.nth_frame,
                                        sliding_step=self.sliding_step)
        self.val_set = GasDataSet(self.val_dir, 
                                      seq_len=self.seq_len, 
                                      nth_frame=self.nth_frame,
                                      sliding_step=self.sliding_step)
        self.test_set = GasDataSet(self.test_dir, 
                                      seq_len=self.seq_len, 
                                      nth_frame=self.nth_frame,
                                      sliding_step=self.sliding_step)
        
    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)