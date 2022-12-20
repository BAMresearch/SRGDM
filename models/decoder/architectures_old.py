#from convlstm import ConvLSTM
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import torchvision.transforms as transforms


class DecoderNet(nn.Module):
    def __init__(self, inner_dims, seq_len=1):
        super().__init__() 
        self.inner_dims = inner_dims
        
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(seq_len, inner_dims[0], kernel_size=(2), stride=1, padding=0),  # [c,7,6]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[0], inner_dims[1], kernel_size=(3), stride=1, padding=0),       # [c,9,8]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[1], inner_dims[2], kernel_size=(3), stride=1, padding=0),       # [c,12,11]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[2], inner_dims[3], kernel_size=(3), stride=1, padding=(0)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[3], inner_dims[4], kernel_size=(4), stride=1, padding=(0,1)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[4], 1, kernel_size=(4,3), stride=2, padding=(2,1)),     # [c,30,25]
        )
    
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

"""
class SequentialDecoder(nn.Module):
    def __init__(self, inner_dims, seq_len=1):
        super().__init__() 
        self.inner_dims = inner_dims
        
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(seq_len, inner_dims[0], kernel_size=(2), stride=1, padding=0),  # [c,7,6]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[0], inner_dims[1], kernel_size=(3), stride=1, padding=0),       # [c,9,8]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[1], inner_dims[2], kernel_size=(3), stride=1, padding=0),       # [c,12,11]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[2], inner_dims[3], kernel_size=(3), stride=1, padding=(0)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[3], inner_dims[4], kernel_size=(4), stride=1, padding=(0,1)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[4], 1, kernel_size=(4,3), stride=2, padding=(2,1)),     # [c,30,25]
        )
    
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded    
"""
### CONVLSTM
    
class decoderNet(nn.Module):
    def __init__(self, input_dim, inner_dims):
        super().__init__() 
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(input_dim, inner_dims[0], kernel_size=(3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[0], inner_dims[1], kernel_size=(3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[1], inner_dims[2], kernel_size=(3), stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[2], inner_dims[3], kernel_size=(3), stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[3], inner_dims[4], kernel_size=(3), stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[4], 1, kernel_size=(4,3), stride=1, padding=(0,0)),
        )
    
    def forward(self, X):
        decoded = self.decoder(X)
        return decoded

class convlstmNet(nn.Module):
    def __init__(self, seq_len, inner_dims):
        super().__init__()
        self.inner_dims = inner_dims
        self.convlstm = ConvLSTM(input_dim=1,
                                 hidden_dim=inner_dims,
                                 kernel_size=(3, 3),
                                 num_layers=len(inner_dims),
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
    def forward(self, X):
        """
        input: [batch, timestep, channel, height, width]
        """ 
        pred, hidden = self.convlstm(X)
        # The output of convlstm is a list containing one tensor. Only return the tensor:
        return pred[0]
    
    
class Net(nn.Module):
    def __init__(self, seq_len, inner_dims_convlstm, inner_dims_decoder):
        super().__init__()
        self.inner_dims_convlstm = inner_dims_convlstm
        self.inner_dims_decoder = inner_dims_decoder
        self.convlstmNet = convlstmNet(seq_len, inner_dims_convlstm)
        self.decoderNet = decoderNet(seq_len, inner_dims_decoder)
        
    def forward(self, X):
        """
        input: [
                gas      [batch, timestep, channel, height, width],
                ]  
        """    
        pred = self.convlstmNet(X)
        # Remove channel dim
        pred = pred.squeeze(2)
        
        # Put concatenation of the ConvLSTM's prediction and current measurements 
        # into the decoder network
        pred = self.decoderNet(pred)

        return pred