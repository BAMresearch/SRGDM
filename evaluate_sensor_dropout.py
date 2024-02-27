
from tqdm import tqdm
import torch
from torch.utils import data
import scipy.interpolate as interpolate
import numpy as np
import multiprocessing

import gdm_metrics
from data.gdm_dataset import GasDataSet
from models.decoder import architectures
from models.kernel_dmv.my_kernel_dmv import KernelDMV
from models.gmrf.my_gmrf import myGMRF

# Load models
decoder = architectures.LightningDecoderNet.load_from_checkpoint("logs/bestparams/version_0/checkpoints/epoch=49-step=75900.ckpt")
decoder.eval()
decoder_denoise = architectures.LightningDecoderNet.load_from_checkpoint("logs/bestparams_noise/version_1/checkpoints/epoch=49-step=75900.ckpt")
decoder_denoise.eval()
kdm = KernelDMV()
gmrf = myGMRF()

# Define models dictionary
models = {
    "decoder": decoder,
    "decoder_denoise": decoder_denoise,
    "gmrf": gmrf,
    "kdm": kdm
}

def interpolate_batch(X, dropout_probability=0.5):
    """Interpolate a batch of size [batch, channel, width, height]"""
    all_interpolated = torch.empty([1,6,5])
    
    for sample in range(X.shape[0]):
        this_X = X[sample].clone()
        
        r = np.linspace(0, 1, this_X.shape[0])
        c = np.linspace(0, 1, this_X.shape[1])

        rr, cc = np.meshgrid(c, r)
        mask = this_X.bernoulli(1-dropout_probability).bool()
        
        try:
            f = interpolate.Rbf(rr[mask], cc[mask], this_X[mask], function='linear')
            interpolated = torch.tensor(f(rr, cc)).unsqueeze(0).float()
        except:
            interpolated = this_X   

        all_interpolated = torch.cat([all_interpolated, interpolated])
        
    all_interpolated = all_interpolated.unsqueeze(1).unsqueeze(1)
    return all_interpolated[1:]

def run_test(dropout_probability, dataset):
    rmse = {"decoder": 0, "decoder_denoise": 0, "gmrf": 0, "kdm": 0}
    kld = {"decoder": 0, "decoder_denoise": 0, "gmrf": 0, "kdm": 0}
    dataloader = iter(data.DataLoader(dataset, shuffle=False, drop_last=True))
    for X, y in tqdm(dataloader):
        count += 1
        X = X.squeeze(1)  # Assuming models expect this shape
        with torch.no_grad():
            for model_name in models:
                if "decoder" in model_name:
                    y_pred = models[model_name](X.squeeze(1).to('cuda:3')).to('cpu')
                elif model_name in ["gmrf", "kdm"]:
                    y_pred = models[model_name].calculate(X.squeeze())[None, None, :]
                else:
                    continue  # Skip if model not recognized
                            
                # Calculate and accumulate metrics
                rmse[model_name] += gdm_metrics.rmse(y_pred, y)
                kld[model_name] += gdm_metrics.kld(y_pred, y)
    return rmse, kld

def run_test_wrapper(args):
    dropout_probability, dataset_path = args
    dataset = GasDataSet(dataset_path)  # Initialize dataset here to ensure it's done in each process
    return run_test(dropout_probability, dataset)

def write_results(dropout_probability, rmse, kld):
    with open(f"results/dropout_p_{dropout_probability}.txt", "w") as f:
        for model_name in rmse.keys():
            f.write(f"{model_name} RMSE: {rmse[model_name]}, KLD: {kld[model_name]}\n")

if __name__ == '__main__':
    dropout_probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
    dataset_path = "data/30x25/test.pt"

    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(run_test_wrapper, [(dp, dataset_path) for dp in dropout_probabilities])

    for i, dropout_probability in enumerate(dropout_probabilities):
        rmse, kld = results[i]
        write_results(dropout_probability, rmse, kld)