from tqdm import tqdm
import gdm_metrics
from torch.utils import data
from data.gdm_dataset import GasDataSet
import torch
from models.decoder import architectures
from models.kernel_dmv.my_kernel_dmv import KernelDMV
from models.gmrf.my_gmrf import myGMRF

# Load models as before
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

dataset = GasDataSet("data/30x25/test.pt", noise=True)
dataloader = iter(data.DataLoader(dataset, shuffle=False, drop_last=True))

rmse = {model: 0 for model in models}
kld = {model: 0 for model in models}
timing = {model: 0 for model in models}  # Dictionary to keep track of timing
count = 0  # to keep track of number of batches for averaging

model_device = decoder.device

for X, y in tqdm(dataloader):
    count += 1
    X = X.squeeze(1)  # Assuming models expect this shape
    with torch.no_grad():
        for model_name in models:
            start_time = time.time()  # Start timing
            if "decoder" in model_name:
                y_pred = models[model_name](X.squeeze(1).to(model_device)).to('cpu')
            elif model_name in ["gmrf", "kdm"]:
                y_pred = models[model_name].calculate(X.squeeze())[None, None, :]
            else:
                continue  # Skip if model not recognized
                        
            # Calculate and accumulate metrics
            rmse[model_name] += gdm_metrics.rmse(y_pred, y)
            kld[model_name] += gdm_metrics.kld(y_pred, y)


# Average the metrics over all batches
for model in models:
    rmse[model] /= count
    kld[model] /= count

# Save the metrics and timing to a text file
with open("results/noisy_test_set_speedup.txt", "w") as f:
    for model_name in models:
        f.write(f"{model_name} RMSE: {rmse[model_name]}, KLD: {kld[model_name]}, Time: {timing[model_name]:.2f} seconds\n")