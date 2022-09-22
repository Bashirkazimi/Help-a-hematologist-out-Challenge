"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
import numpy as np
import copy
import torchvision
from torchvision import transforms
import random


crop_Ace20=250
crop_Mat19=345
crop_WBC1=288

dataset_image_size = {
    "Ace_20": crop_Ace20,   #250,
    "Mat_19": crop_Mat19,   #345,
    "WBC1": crop_WBC1,   #288,
}


class DatasetGenerator(Dataset):

    def __init__(self,
                 df_source,
                 df_target,
                 transform=None,
                 selected_channels=[0, 1, 2]
                 ):

        self.df_source = df_source.copy().reset_index(drop=True)
        self.df_target = df_target.copy().reset_index(drop=True)
        self.transform = transform
        self.selected_channels = selected_channels

    def __len__(self):
        return max(len(self.df_source), len(self.df_target))

    def __getitem__(self, idx):

        source_index = idx % len(self.df_source)
        ## get image and label
        dataset = self.df_source.loc[source_index, "dataset"]
        source_name = self.df_source.loc[source_index, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.df_source.loc[source_index, "file"]
        source_path = h5_file_path
        image = imread(h5_file_path)[:, :, self.selected_channels]
        image = (image / 255.) * 2 - 1
        h1 = (image.shape[0] - crop_size) / 2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) / 2
        h2 = int(h2)

        w1 = (image.shape[1] - crop_size) / 2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) / 2
        w2 = int(w2)
        image = image[h1:h2, w1:w2, :]
        image = np.transpose(image, (2, 0, 1))


        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        source_image = image.float()

        if self.transform:
            source_image = self.transform(source_image)

        target_index = random.randint(0, len(self.df_target) - 1)

        dataset = self.df_target.loc[target_index, "dataset"]
        target_name = self.df_target.loc[target_index, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.df_target.loc[target_index, "file"]
        target_path = h5_file_path
        image = imread(h5_file_path)[:, :, self.selected_channels]
        image = (image / 255.) * 2 - 1
        h1 = (image.shape[0] - crop_size) / 2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) / 2
        h2 = int(h2)

        w1 = (image.shape[1] - crop_size) / 2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) / 2
        w2 = int(w2)
        image = image[h1:h2, w1:w2, :]
        image = np.transpose(image, (2, 0, 1))


        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        target_image = image.float()

        if self.transform:
            target_image = self.transform(target_image)

        return {'A': source_image.float(), 'B': target_image.float(), 'A_paths': source_path, 'B_paths': target_path}


class DatasetGeneratorMeanStd(Dataset):

    def __init__(self,
                 df_source,
                 df_target,
                 transform=None,
                 selected_channels=[0, 1, 2]
                 ):

        self.df_source = df_source.copy().reset_index(drop=True)
        self.df_target = df_target.copy().reset_index(drop=True)
        self.transform = transform
        self.selected_channels = selected_channels

    def __len__(self):
        return max(len(self.df_source), len(self.df_target))

    def __getitem__(self, idx):

        source_index = idx % len(self.df_source)
        ## get image and label
        dataset = self.df_source.loc[source_index, "dataset"]
        source_name = self.df_source.loc[source_index, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.df_source.loc[source_index, "file"]
        source_path = h5_file_path
        image = imread(h5_file_path)[:, :, self.selected_channels]
        image = image / 255.
        h1 = (image.shape[0] - crop_size) / 2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) / 2
        h2 = int(h2)

        w1 = (image.shape[1] - crop_size) / 2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) / 2
        w2 = int(w2)
        image = image[h1:h2, w1:w2, :]
        image = np.transpose(image, (2, 0, 1))


        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        source_image = image.float()

        if self.transform:
            source_image = self.transform(source_image)

        target_index = random.randint(0, len(self.df_target) - 1)

        dataset = self.df_target.loc[target_index, "dataset"]
        target_name = self.df_target.loc[target_index, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.df_target.loc[target_index, "file"]
        target_path = h5_file_path
        image = imread(h5_file_path)[:, :, self.selected_channels]
        image = image / 255.
        h1 = (image.shape[0] - crop_size) / 2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) / 2
        h2 = int(h2)

        w1 = (image.shape[1] - crop_size) / 2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) / 2
        w2 = int(w2)
        image = image[h1:h2, w1:w2, :]
        image = np.transpose(image, (2, 0, 1))


        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        target_image = image.float()

        if self.transform:
            target_image = self.transform(target_image)

        return {'A': source_image.float(), 'B': target_image.float(), 'A_paths': source_path, 'B_paths': target_path}


class InputDatasetGenerator(Dataset):

    def __init__(self,
                 df,
                 transform=None,
                 selected_channels=[0, 1, 2]
                 ):

        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        self.selected_channels = selected_channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        source_index = idx % len(self.df)
        ## get image and label
        dataset = self.df.loc[source_index, "dataset"]
        source_name = self.df.loc[source_index, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.df.loc[source_index, "file"]
        source_path = h5_file_path
        image = imread(h5_file_path)[:, :, self.selected_channels]
        image = (image / 255.) * 2 - 1
        h1 = (image.shape[0] - crop_size) / 2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) / 2
        h2 = int(h2)

        w1 = (image.shape[1] - crop_size) / 2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) / 2
        w2 = int(w2)
        image = image[h1:h2, w1:w2, :]
        image = np.transpose(image, (2, 0, 1))


        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        source_image = image.float()

        if self.transform:
            source_image = self.transform(source_image)

        return {'A': source_image.float(), 'B': source_image.float(), 'A_paths': source_path,  'B_paths': source_path}


class InputDatasetGeneratorMeanStd(Dataset):

    def __init__(self,
                 df,
                 transform=None,
                 selected_channels=[0, 1, 2]
                 ):

        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        self.selected_channels = selected_channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        source_index = idx % len(self.df)
        ## get image and label
        dataset = self.df.loc[source_index, "dataset"]
        source_name = self.df.loc[source_index, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.df.loc[source_index, "file"]
        source_path = h5_file_path
        image = imread(h5_file_path)[:, :, self.selected_channels]
        image = image / 255.
        h1 = (image.shape[0] - crop_size) / 2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) / 2
        h2 = int(h2)

        w1 = (image.shape[1] - crop_size) / 2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) / 2
        w2 = int(w2)
        image = image[h1:h2, w1:w2, :]
        image = np.transpose(image, (2, 0, 1))


        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        source_image = image.float()

        if self.transform:
            source_image = self.transform(source_image)

        return {'A': source_image.float(), 'B': source_image.float(), 'A_paths': source_path,  'B_paths': source_path}


def get_input_dataset(csv_file, batch_size=32, num_workers=4, inputWBC=True):
    resize = 224  # image pixel size

    random_crop_scale = (0.8, 1.0)
    random_crop_ratio = (0.8, 1.2)

    # mean = [0.485, 0.456, 0.406]  # values from imagenet
    # std = [0.229, 0.224, 0.225]  # values from imagenet

    # normalization = torchvision.transforms.Normalize(mean, std)

    transform = transforms.Compose([
        # normalization,
        transforms.Resize(resize)])

    df = pd.read_csv(csv_file, index_col=None)
    # filter out the DEV VALID WBC1 dataset
    if inputWBC:
        df = df[df.dataset == 'WBC1']
    else:
        df = df[df.dataset != 'WBC1']

    dataset = InputDatasetGenerator(df, transform=transform)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader


def get_input_dataset_mean_std(csv_file, batch_size=32, num_workers=4, inputWBC=True):
    resize = 224  # image pixel size

    with open('Datasets/Mean_image.pickle', 'rb') as f:
        mean_im = pickle.load(f)
    with open('Datasets/Std_image.pickle', 'rb') as f:
        std_im = pickle.load(f)

    ####
    normalization = transforms.Lambda(lambda im: (im - mean_im) / std_im)

    transform = transforms.Compose([
        transforms.Resize(resize),
        normalization])

    df = pd.read_csv(csv_file, index_col=None)
    # filter out the DEV VALID WBC1 dataset
    if inputWBC:
        df = df[df.dataset == 'WBC1']
    else:
        df = df[df.dataset != 'WBC1']

    dataset = InputDatasetGeneratorMeanStd(df, transform=transform)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader


def get_source_and_target(csv_file, batch_size=32, num_workers=4):
    resize = 224  # image pixel size

    random_crop_scale = (0.8, 1.0)
    random_crop_ratio = (0.8, 1.2)

    # mean = [0.485, 0.456, 0.406]  # values from imagenet
    # std = [0.229, 0.224, 0.225]  # values from imagenet

    # normalization = torchvision.transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        # normalization,
        transforms.RandomResizedCrop(resize, scale=random_crop_scale, ratio=random_crop_ratio),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    val_transform = transforms.Compose([
        # normalization,
        transforms.Resize(resize)])

    df = pd.read_csv(csv_file, index_col=None)
    # filter out the DEV VALID WBC1 dataset
    df_source = df[df.dataset != 'WBC1']
    df_target = df[df.dataset == 'WBC1']

    # split the dataset, getting equal percentages of labels and equal percentages of Mat_19 and Ace_20 datasets
    train, test = train_test_split(df_source, test_size=0.2, random_state=42, stratify=df_source[['label', 'dataset']])
    train, val = train_test_split(train, test_size=len(test), random_state=42, stratify=train[['label', 'dataset']])

    train_dataset = DatasetGenerator(train, df_target, transform=train_transform)
    val_dataset = DatasetGenerator(val, df_target, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, valid_loader


def get_source_and_target_mean_std(csv_file, batch_size=32, num_workers=4):
    resize = 224  # image pixel size

    resize = 224  # image pixel size

    with open('Datasets/Mean_image.pickle', 'rb') as f:
        mean_im = pickle.load(f)
    with open('Datasets/Std_image.pickle', 'rb') as f:
        std_im = pickle.load(f)

    ####
    normalization = transforms.Lambda(lambda im: (im - mean_im) / std_im)

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        normalization,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize),
        normalization])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        normalization])

    df = pd.read_csv(csv_file, index_col=None)
    # filter out the DEV VALID WBC1 dataset
    df_source = df[df.dataset != 'WBC1']
    df_target = df[df.dataset == 'WBC1']

    # split the dataset, getting equal percentages of labels and equal percentages of Mat_19 and Ace_20 datasets
    train, test = train_test_split(df_source, test_size=0.2, random_state=42, stratify=df_source[['label', 'dataset']])
    train, val = train_test_split(train, test_size=len(test), random_state=42, stratify=train[['label', 'dataset']])

    train_dataset = DatasetGeneratorMeanStd(train, df_target, transform=train_transform)
    val_dataset = DatasetGeneratorMeanStd(val, df_target, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, valid_loader


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
