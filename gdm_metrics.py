import torch
import torch.nn as nn

def rmse(pred_dist, true_dist):
    """ Calculate root mean squared error."""
    return torch.sqrt(nn.functional.mse_loss(pred_dist, true_dist))
    
def kld(pred_dist, true_dist):
    """ Calculate Kullback-Leibler divergence."""
    softmax = nn.LogSoftmax(dim=2)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    return kl_loss(softmax(pred_dist.contiguous().view(1,1,-1)), softmax(true_dist.view(1,1,-1)))


