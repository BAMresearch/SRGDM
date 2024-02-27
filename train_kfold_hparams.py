import optuna
import numpy as np
import torch
import pickle


import logging
np.random.seed(1991)
torch.manual_seed(1991)


import pytorch_lightning as pl
from models.decoder.architectures import LightningDecoderNet
from data.gdm_dataset_kfold import create_fold_indices, GasDataModule


logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


BATCH_SIZE = 32
EPOCHS = 30
NUM_WORKERS = 10 # number of CPUs used
NTH_FRAME = 2
SLIDING_STEP = 1
SEQ_LEN = 1
DIR = "data/30x25"

def objective(trial):
    num_layers = 5
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2)
    inner_dims = [trial.suggest_int("inner_dims_{}".format(i), 3, 256, log=True) for i in range(num_layers)]

    trial_number = trial.number

    #total_val_loss = 0
    fold_val_losses = [] 

    for fold, indices in enumerate(fold_indices):
        model = LightningDecoderNet(inner_dims, SEQ_LEN, learning_rate)
        datamodule = GasDataModule(data_dir=DIR, seq_len=SEQ_LEN, nth_frame=NTH_FRAME, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                                   fold_indices=indices, fold_index=fold)
        
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=f"logs_kfold", 
            name=f"trial_{trial_number}",
            version=f"fold_{fold}"
        )
        trainer = pl.Trainer(
            logger = tb_logger,
            min_epochs = int(EPOCHS/2),
            max_epochs = EPOCHS,
            accelerator="auto" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=True,
            enable_model_summary=False,
            devices=[3],
            fast_dev_run=False # set to False, if model should be trained
        )
        
        trainer.fit(model, datamodule=datamodule)

        # Collect validation loss for each fold
        fold_val_loss = trainer.callback_metrics["val_loss"].item()
        fold_val_losses.append(fold_val_loss)

        # Log validation loss for each fold
        tb_logger.experiment.add_scalar("Fold_val_loss", fold_val_loss, global_step=fold)
        

    # Calculate average and variance of validation loss across all folds
    average_val_loss = np.mean(fold_val_losses)
    val_loss_variance = np.var(fold_val_losses)

    # Log average validation loss and variance at the end of all epochs (across folds)
    tb_logger.experiment.add_scalar("Aggregate Metrics/average_val_loss", average_val_loss, global_step=trial_number)
    tb_logger.experiment.add_scalar("Aggregate Metrics/val_loss_variance", val_loss_variance, global_step=trial_number)

    return average_val_loss


optuna.logging.set_verbosity(optuna.logging.ERROR)

n_simulations = 120  # 30 sources * 12 simulations each = 360
n_folds = 10
fold_indices = create_fold_indices(n_simulations, n_folds)

starting_params={"lr":5e-4}

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

# After the optimization process
study_name = "decoder_study.pkl"  # Name of the file to save the study
with open(study_name, "wb") as f:
    pickle.dump(study, f)

print(f"Study saved to {study_name}")



