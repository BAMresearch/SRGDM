import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import torchvision.transforms as transforms
import pytorch_lightning as pl

##--------
# DECODER

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
    
class LightningNet(pl.LightningModule):
    def __init__(self, inner_dims, seq_len, learning_rate):
        super().__init__()        
        self.model = DecoderNet(inner_dims=inner_dims, seq_len=seq_len)
        self.inner_dims = inner_dims
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch  
        # .squeeze(2) to remove the seq_len dim:
        # [batch, seq_len, channel, width, height] -> [batch, seq_len, width, height]
        y_hat = self(X.squeeze(1)) 
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch  
        y_hat = self(X.squeeze(2))
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X.squeeze(2))
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log("loss", {"loss":loss, "rmse": rmse})
        
        
        
## --------
# CONVLSTM
from models.decoder.convlstm import ConvLSTM        
        
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
    
class ConvLSTMDecoder(nn.Module):
    def __init__(self, seq_len, inner_dims_convlstm, inner_dims_decoder):
        super().__init__()
        self.inner_dims_convlstm = inner_dims_convlstm
        self.inner_dims_decoder = inner_dims_decoder
        self.convlstm = ConvLSTM(input_dim=1,
                                 hidden_dim=inner_dims_convlstm,
                                 kernel_size=(3, 3),
                                 num_layers=len(inner_dims),
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.decoder = DecoderNet(inner_dims_decoder, seq_len)
        
    def forward(self, X):
        """
        input: [
                gas      [batch, timestep, channel, height, width],
                ]  
        """    
        pred = self.convlstm(X)
        # Remove channel dim
        pred = pred.squeeze(2)
        
        # Put concatenation of the ConvLSTM's prediction and current measurements 
        # into the decoder network
        pred = self.decoderNet(pred)

        return pred
    
    
class DecoderConvLSTM(nn.Module):
    """ Takes a pre-trained decoder and applies ConvLSTM to it."""
    def __init__(self, seq_len, inner_dims_convlstm, decoder):
        super().__init__()
        self.seq_len = seq_len
        
        self.inner_dims_convlstm = inner_dims_convlstm
        self.convlstm = ConvLSTM(input_dim=1,
                                 hidden_dim=inner_dims_convlstm,
                                 kernel_size=(3, 3),
                                 num_layers=len(inner_dims_convlstm),
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.decoder = decoder
        
    def forward(self, X):
        """
        input: [
                gas      [batch, timestep, channel, height, width],
                ]  
        """    
        # Upscale each img
        pred = torch.zeros([X.shape[0], 10, 1, 30, 25])

        for batch in range(X.shape[0]):
            pred[batch] = decoder(X[batch])
        
        #pred = self.decoderNet(pred)
        pred, hidden = self.convlstm(pred)
        pred = pred[0]

        return pred