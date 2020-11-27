import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split

import os
from PIL import Image
import random

from sklearn.utils import shuffle
import numpy as np


img_size = 256
root_path_data = "Data/img_train_shape"
root_path_labels = "Data/img_train_skeleton_grayscale"


def get_training_data(seed=1):
    file_list = os.listdir(root_path_data)

    trn_list, vld_list = train_val_split(file_list, r_seed=seed)

    trn_dataset = ImageDataset(trn_list)
    vld_dataset = ImageDataset(vld_list)

    return trn_dataset, vld_dataset

# Input: a list of all filenames
# Output: one list with filenames for training and another for validation


def train_val_split(file_list, r_seed=1):
    # random shuffle the filenames
    file_list = shuffle(file_list, random_state=r_seed)

    # amount of validation data, e.g 0.1 = 10%
    val_split = 0.1
    # amount of training data
    train_split = 1-val_split

    # split into a train and validation set
    n = len(file_list)
    split_at = int(np.floor(n*train_split))
    train = file_list[:split_at]
    val = file_list[split_at:]

    return train, val


def get_random_transform(p_padding=0.5, p_vflip=0.5, p_hflip=0.5, p_rotate=0.5):
    transform_list = []

    padding = random.uniform(0, 1) < p_padding
    vflip = random.uniform(0, 1) < p_vflip
    hflip = random.uniform(0, 1) < p_hflip
    rotate = random.uniform(0, 1) < p_rotate

    def rand_int():
        return random.randint(1, 20)

    if padding:
        padding_tuple = (rand_int(), rand_int(), rand_int(), rand_int())
        transform_list.append(transforms.Pad(padding_tuple))
        transform_list.append(transforms.CenterCrop(img_size))

    if vflip:
        transform_list.append(transforms.functional.vflip)

    if hflip:
        transform_list.append(transforms.functional.hflip)

    if rotate:
        angle = random.choice([-90, 90])
        transform_list.append(
            transforms.RandomRotation(degrees=[angle, angle]))

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


# Dataset for handling the image input and output.
# Takes the directory to the input and output files.
# Requires the input and output files to have the same names.
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
        self.trn_dir = root_path_data
        self.lbl_dir = root_path_labels
        self.file_list = file_list
        # self.file_list.remove(".DS_Store")

    def __getitem__(self, idx):
        transform = get_random_transform()

        # Load the img and turn it into a Torch tensor matrix
        link = self.trn_dir+"/"+self.file_list[idx]
        data = transform(Image.open(link))

        # Create label
        link = self.lbl_dir+"/"+self.file_list[idx]
        label = transform(Image.open(link))

        return (data, label)

    def __len__(self):
        return len(self.file_list)
