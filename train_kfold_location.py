# Dataset
DIR = "data/30x25/"
SEQ_LEN = 1
NTH_FRAME = 2

# Model
INNER_DIMS = [62, 157, 255, 62, 37]
LEARNING_RATE = 0.0015434050393851495

# Training
BATCH_SIZE = 64
NUM_WORKERS = 10
EPOCHS = 50

import torch
import numpy as np
import pytorch_lightning as pl
from models.decoder.architectures import LightningDecoderNet
from data.gdm_dataset_kfold import create_fold_indices, GasDataModule

fold_dimension = 0
fold_indices = create_fold_indices(n_simulations=30, n_folds=10)


fold_val_losses = [] 
for fold, indices in enumerate(fold_indices):
    model = LightningDecoderNet(INNER_DIMS, SEQ_LEN, LEARNING_RATE)
    datamodule = GasDataModule(data_dir=DIR, seq_len=SEQ_LEN, nth_frame=NTH_FRAME, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                               fold_dimension=fold_dimension, fold_indices=fold_indices, fold_index=fold)
    
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=f"logs_kfold_scenarios", 
        name=f"locations",
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
        fast_dev_run=False
    )
    
    trainer.fit(model, datamodule=datamodule)

    # Collect validation loss for each fold
    fold_val_loss = trainer.callback_metrics["val_loss"].item()
    fold_val_losses.append(fold_val_loss)

    # Log validation loss for each fold
    tb_logger.experiment.add_scalar("Fold_val_loss", fold_val_loss, global_step=fold)

# Convert the fold validation losses to a numpy array for statistical calculations
fold_val_losses_np = np.array(fold_val_losses)

# Calculate mean and standard deviation
mean_val_loss = np.mean(fold_val_losses_np)
std_val_loss = np.std(fold_val_losses_np)

# Print the summary statistics
print(f"Validation Loss across {len(fold_indices)} folds: Mean = {mean_val_loss:.4f}, StdDev = {std_val_loss:.4f}")