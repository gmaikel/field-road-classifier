from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from module import FieldsRoadsModule
from datamodules import FieldsRoadsDataModule

import random
import numpy as np
import torch

seed_value = 1
# Set Python random seed
random.seed(seed_value)
# Set Numpy random seed
np.random.seed(seed_value)
# Set PyTorch random seed
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True  # if using GPU, ensure reproducibility
torch.backends.cudnn.benchmark = False


if __name__=='__main__':
    torch.set_float32_matmul_precision('high')  # ou 'medium'

    # Instantiate the model
    module = FieldsRoadsModule()
    fields_roads_dm = FieldsRoadsDataModule(
        train_dir='/home/mgali/PycharmProjects/trimble/data/dataset/train',
        val_dir='/home/mgali/PycharmProjects/trimble/data/dataset/val',
        batch_size=16,
    )

    # Instantiate an EarlyStopping callback to prevent overfitting
    early_stopping_callback = EarlyStopping('val_loss', patience=100, verbose=False, mode='min')

    # Instantiate a TensorBoard logger to log training and validation metrics
    logger = TensorBoardLogger('tb_logs', name='field_roads_model')

    # Instantiate a Trainer object
    trainer = Trainer(accelerator="gpu", devices="auto",
                      max_epochs=200, callbacks=[early_stopping_callback], logger=logger,
                      )

    # Train the model
    trainer.fit(model=module, datamodule=fields_roads_dm)