import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss

# Local
from models import FieldsRoadsModel
from losses import FocalLoss


# class FieldsRoadsModule(pl.LightningModule):
#     def __init__(self, learning_rate=1e-3, loss_func='FocalLoss'):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.model = FieldsRoadsModel()
#
#     def forward(self, x):
#         return self.model(x)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         if self.loss_func == 'FocalLoss':
#             loss = FocalLoss(loss_fcn=BCEWithLogitsLoss())(y_hat.squeeze(), y.float())
#         else:
#             loss = BCEWithLogitsLoss()(y_hat.squeeze(), y.float())
#         y_pred = (y_hat > 0.5).int()
#         acc = (y_pred == y).float().mean()
#         self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         if self.loss_func == 'FocalLoss':
#             loss = FocalLoss(loss_fcn=BCEWithLogitsLoss())(y_hat.squeeze(), y.float())
#         else:
#             loss = BCEWithLogitsLoss()(y_hat.squeeze(), y.float())
#         y_pred = (y_hat > 0.5).int()
#         acc = (y_pred == y).float().mean()
#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#     def configure_optimizers(self):
#         optimizer = optim.Adam(
#             self.parameters(),
#             # lr=self.learning_rate,
#         )
#         return optimizer
#

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class FieldsRoadsModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 1)
        self.loss = torch.nn.functional.hinge_embedding_loss

    def forward(self, x):
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.squeeze(), y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.squeeze(), y.float())
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer