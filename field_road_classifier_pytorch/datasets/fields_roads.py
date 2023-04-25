import glob
import cv2
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
import os
from torchvision import datasets, transforms

from torchvision import datasets


# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(224, 224)),
    # Flip the images randomly on the horizontal
    # transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

train_data = datasets.ImageFolder(root='/home/mgali/PycharmProjects/trimble/data/dataset/train', # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root='/home/mgali/PycharmProjects/trimble/data/dataset/val',
                                 transform=data_transform)






if __name__=='__main__':
    # Setup batch size and number of workers
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")