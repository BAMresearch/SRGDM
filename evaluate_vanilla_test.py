from tqdm import tqdm
import gdm_metrics
from torch.utils import data
from data.gdm_dataset import GasDataSet
from evaluate import evaluate_models

###############
# LOAD MODELS #
###############

# ~~~~~~~~~~~~~~~~
# PyTorch Model

import torch
from models.decoder import architectures

decoder = architectures.LightningDecoderNet.load_from_checkpoint("logs/bestparams/version_0/checkpoints/epoch=49-step=75900.ckpt").to('cpu')
decoder.eval();

decoder_denoise = architectures.LightningDecoderNet.load_from_checkpoint("logs/bestparams_noise/version_0/checkpoints/epoch=49-step=75900.ckpt").to('cpu')
decoder_denoise.eval();

# ~~~~~~~~~~~~~~~~
# Kernel DM+V

from models.kernel_dmv.my_kernel_dmv import KernelDMV
kdm = KernelDMV()

# ~~~~~~~~~~~~~~~~
# GMRF

from models.gmrf.my_gmrf import myGMRF
gmrf = myGMRF()


####################
# VANILLA TEST SET #
####################

# Define models dictionary
models = {
    "decoder": decoder,
    "decoder_denoise": decoder_denoise,
    "gmrf": gmrf,
    "kdm": kdm
}

dataset = GasDataSet("data/30x25/test.pt")
dataloader = iter(data.DataLoader(dataset, shuffle=False, drop_last=True))

# Evaluate models
rmse, kld = evaluate_models(dataloader, models)

# Save the metrics to a text file
with open("results/vanilla_test_set.txt", "w") as f:
    for model_name in models:
        f.write(f"{model_name} RMSE: {rmse[model_name]}, KLD: {kld[model_name]}\n")