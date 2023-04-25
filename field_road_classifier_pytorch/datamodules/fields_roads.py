from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl


class FieldsRoadsDataModule(pl.LightningDataModule):
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
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            T.Lambda(lambda x: x + 0.5)  # Shift values from -0.5 to 0.5 to 0 to 1
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