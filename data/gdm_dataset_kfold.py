import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import KFold

def create_fold_indices(n_simulations, n_folds=10):
    """
    Create fold indices for k-fold cross-validation.

    Parameters:
    - n_simulations (int): The total number of simulations.
    - n_folds (int): The number of folds for cross-validation.

    Returns:
    - A list of tuples, where each tuple contains two arrays: 
      the first array for training indices and the second for validation indices.
    """
    kf = KFold(n_folds, shuffle=True, random_state=42)
    indices = np.arange(n_simulations)
    folds = list(kf.split(indices))
    
    # Convert to list of tuples for easier handling
    fold_indices = [(train_idx, test_idx) for train_idx, test_idx in folds]
    return fold_indices

class GasDataSet(data.Dataset):   
    """
    Dataset that returns a sequence of gas distribution data.
    Output:
        X - Masked gas distribution data of length `seq_len` of time step T and the preceeding time steps.
        y - Gas distribution of time step T.
    """
    def __init__(self, root, seq_len=1, nth_frame=2, sliding_step=1, fold_dimension=None, fold_indices=None, fold_index=None, for_training=True):
        self.root = root
        self.seq_len = seq_len
        
        # Load the data. The data tensor is structured: [n_sources, n_simulations, n_images, height, width]
        # n_sources -> individual gas source position
        # n_simulations -> individual simulations of each source
        # n_images  ->  images of each simulation run 
        x = torch.load(root)        

        #~~~~~~~~~~~~~~~~~
        # Start fold selection
        # Check if fold_dimension, fold_indices, and fold_index are provided
        if fold_indices is not None and fold_index is not None:
            if fold_dimension is not None:
                assert fold_dimension in [0, 1], "fold_dimension must be either 0 (n_sources) or 1 (n_simulations)"
                train_idx, val_idx = fold_indices[fold_index]

                if fold_dimension == 0:
                    # Folding along the first dimension (n_sources)
                    if for_training:
                        x = x[train_idx]  # Use train_idx to filter sources for training
                    else:
                        x = x[val_idx]  # Use val_idx to filter sources for validation
                elif fold_dimension == 1:
                    # Folding along the second dimension (n_simulations)
                    if for_training:
                        x = x[:, train_idx]  # Use train_idx to filter simulations for training
                    else:
                        x = x[:, val_idx]  # Use val_idx to filter simulations for validation
                n_timesteps = x.shape[2]
                x = x.reshape(-1,n_timesteps,30,25)
            else:
                # if no fold_dimension is provided, treat first two dimension equally
                train_idx, val_idx = fold_indices[fold_index]
                n_timesteps = x.shape[2]
                x = x.reshape(-1,n_timesteps,30,25)
                if for_training:
                    x = x[train_idx]
                else:
                    x = x[val_idx] 
        else:
            # For the test set...
            n_timesteps = x.shape[2]
            x = x.reshape(-1,n_timesteps,30,25)
              
        # End fold selection
        #~~~~~~~~~~~~~~~~~            
        
        # Select only the images that we care about (ruled by nth frame)
        n_frames = x.shape[1]
        idxs = torch.tensor(range(n_frames)[::nth_frame])
        x = torch.index_select(x, 1, idxs)

        #~~~~~~~~~~~~~~~~~
        # Start sequencing
        # Get the number of sequences, that each source simulation generates
        n_seq = x[0].unfold(0, seq_len, sliding_step).size()[0]

        # Create an empty tensor that is going to be filled up in the loop
        # Dims: [source simulations, sequences, images_per_sequence, width, height]
        new_x = torch.empty([x.size()[0], n_seq, seq_len, 30, 25], device=x.device, dtype=x.dtype)
        for i in range(x.shape[0]):
            # Use unfold to create sequences then permute dimensions as required.
            new_x[i] = x[i].unfold(0, seq_len, sliding_step).permute(0, 3, 1, 2)
        
        # End sequencing
        #~~~~~~~~~~~~~~~~~
            
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
        return len(self.data)
            
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.unsqueeze(1)
                
        sample = self.transform(sample)

        # Remove channel dimension and add channel dimension before&after downsample
        X = self.downsample(sample.squeeze(1)).unsqueeze(1)
        y = sample[-1]
            
        return X, y
    

class GasDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, seq_len=1, nth_frame=2, sliding_step=1,  num_workers=0, fold_dimension=None, fold_indices=None, fold_index=None):
        super().__init__()
        self.seq_len = seq_len
        self.nth_frame = nth_frame
        self.batch_size = batch_size
        self.sliding_step = sliding_step
        self.num_workers = num_workers
        self.train_dir = os.path.join(data_dir, 'trainvalid.pt')
        self.val_dir = os.path.join(data_dir, 'trainvalid.pt')
        self.test_dir = os.path.join(data_dir, 'test.pt')
        self.fold_dimension = fold_dimension
        self.fold_indices = fold_indices
        self.fold_index = fold_index

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.train_set = GasDataSet(self.train_dir, 
                                            seq_len=self.seq_len, 
                                            nth_frame=self.nth_frame,
                                            sliding_step=self.sliding_step,
                                            fold_dimension=self.fold_dimension,
                                            fold_indices=self.fold_indices,
                                            fold_index=self.fold_index,
                                            for_training=True)
            self.val_set = GasDataSet(self.val_dir, 
                                        seq_len=self.seq_len, 
                                        nth_frame=self.nth_frame,
                                        sliding_step=self.sliding_step,
                                        fold_dimension=self.fold_dimension,
                                        fold_indices=self.fold_indices,
                                        fold_index=self.fold_index,
                                        for_training=True)
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