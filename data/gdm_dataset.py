import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
import pytorch_lightning as pl

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        tensor = torch.clamp(tensor, min=0) # set minimum to 0 (important for gmrf)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GasDataSet(data.Dataset):   
    """
    Dataset that returns a sequence of gas distribution data.
    Output:
        X - Masked gas distribution data of length `seq_len` of time step T and the preceeding time steps.
        y - Gas distribution of time step T.
    """
    def __init__(self, root, seq_len=1, nth_frame=2, sliding_step=1, noise=False):
        self.root = root
        self.seq_len = seq_len        # how many time steps shall be predicted
        self.noise = noise

        # Load the data. The data tensor is structured: [n_sources, n_images, height, width]
        # n_sources ->  individual simulations of sources
        # n_images  ->  images of each simulation run 
        x = torch.load(root)        

        n_timesteps = x.shape[2]
        x = x.reshape(-1,n_timesteps,30,25)

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
        
        if self.noise == True:
            self.apply_noise = AddGaussianNoise(std=0.05)

    
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
        X = self.downsample(sample.squeeze(1))#.unsqueeze(0)
        if self.noise==True:
            X = self.apply_noise(X)
        y = sample[-1]
            
        return X, y
    

class GasDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, seq_len=1, nth_frame=2, sliding_step=1,  num_workers=0, noise=False):
        super().__init__()
        self.seq_len = seq_len
        self.nth_frame = nth_frame
        self.batch_size = batch_size
        self.sliding_step = sliding_step
        self.num_workers = num_workers
        self.train_dir = os.path.join(data_dir, 'trainvalid.pt')
        self.test_dir = os.path.join(data_dir, 'test.pt')
        self.noise = noise
        
    def setup(self, stage = None):
        self.train_set = GasDataSet(self.train_dir, 
                                        seq_len=self.seq_len, 
                                        nth_frame=self.nth_frame,
                                        sliding_step=self.sliding_step,
                                        noise=self.noise)
        self.test_set = GasDataSet(self.test_dir, 
                                      seq_len=self.seq_len, 
                                      nth_frame=self.nth_frame,
                                      sliding_step=self.sliding_step)
        
    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
    
    
    def test_dataloader(self):
        return data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)