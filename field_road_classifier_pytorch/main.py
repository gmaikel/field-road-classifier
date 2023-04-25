import os
import cv2
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

from matplotlib import pyplot as plt

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.functional as tmf
import pytorch_lightning as pl



class BirdDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download or prepare data if necessary
        pass

    def setup(self, stage=None):
        # define image transformations
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # build train and validation datasets
        self.train_dataset = ImageFolder(
            root=self.train_dir, transform=transform
        )
        self.val_dataset = ImageFolder(
            root=self.val_dir, transform=transform
        )

    def train_dataloader(self):
        # return the training dataloader
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        # return the validation dataloader
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class MyModel(pl.LightningModule):
    def __init__(self, conv_channels=32, fc_channels=128, dropout_prob=0.7, learning_rate=1e-3, loss_func='BCEWithLogitsLoss'):
        super().__init__()
        self.conv_channels = conv_channels
        self.fc_channels = fc_channels
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.loss_func = loss_func

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_channels * 56 * 56, fc_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(fc_channels, fc_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(fc_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_layers[0].in_features)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.loss_func == 'FocalLoss':
            loss = FocalLoss()(y_hat.squeeze(), y.float())
        else:
            loss = nn.BCEWithLogitsLoss()(y_hat.squeeze(), y.float())
        y_pred = (y_hat > 0.5).int()
        acc = (y_pred == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.loss_func == 'FocalLoss':
            loss = FocalLoss()(y_hat.squeeze(), y.float())
        else:
            loss = nn.BCEWithLogitsLoss()(y_hat.squeeze(), y.float())
        y_pred = (y_hat > 0.5).int()
        acc = (y_pred == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger

    # Instantiate the model
    model = MyModel(loss_func="FocalLoss")
    bird_dm = BirdDataModule(train_dir='/home/mgali/PycharmProjects/trimble/data/dataset/train',
                             val_dir='/home/mgali/PycharmProjects/trimble/data/dataset/val',
                             batch_size=32)

    # Instantiate an EarlyStopping callback to prevent overfitting
    early_stopping_callback = EarlyStopping('val_loss', patience=20, verbose=False, mode='min')

    # Instantiate a TensorBoard logger to log training and validation metrics
    logger = TensorBoardLogger('tb_logs', name='my_model')

    # Instantiate a Trainer object
    trainer = Trainer(max_epochs=100, callbacks=[early_stopping_callback], logger=logger)

    # Train the model
    trainer.fit(model, bird_dm)
