import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
import numpy as np
import copy
import torchvision
from torchvision import transforms
import pickle
from glob import glob

crop_Ace20=250
crop_Mat19=345
crop_WBC1=288

dataset_image_size = {
    "Ace_20": crop_Ace20,   #250,
    "Mat_19": crop_Mat19,   #345,
    "WBC1": crop_WBC1,   #288,
}

label_map_all = {
        'basophil': 0,
        'eosinophil': 1,
        'erythroblast': 2,
        'myeloblast' : 3,
        'promyelocyte': 4,
        'myelocyte': 5,
        'metamyelocyte': 6,
        'neutrophil_banded': 7,
        'neutrophil_segmented': 8,
        'monocyte': 9,
        'lymphocyte_typical': 10
    }

label_map_reverse = {
        0: 'basophil',
        1: 'eosinophil',
        2: 'erythroblast',
        3: 'myeloblast',
        4: 'promyelocyte',
        5: 'myelocyte',
        6: 'metamyelocyte',
        7: 'neutrophil_banded',
        8: 'neutrophil_segmented',
        9: 'monocyte',
        10: 'lymphocyte_typical'
    }

# The unlabeled WBC dataset gets the classname 'Data-Val' for every image

label_map_pred = {
        'DATA-VAL': 0
    }


class DatasetGenerator(Dataset):

    def __init__(self,
                 metadata,
                 transform=None,
                 selected_channels=[0, 1, 2],
                 return_label=True,
                 fake_dir='Datasets/MAT_ACE_AS_WBC'
                 ):

        self.metadata = metadata.copy().reset_index(drop=True)
        self.transform = transform
        self.selected_channels = selected_channels
        self.return_label = return_label

        self.fake_dir = fake_dir  # 'pytorch-CycleGAN-and-pix2pix/Datasets/MAT_ACE_AS_WBC'

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset = self.metadata.loc[idx, "dataset"]
        image_name = self.metadata.loc[idx, "Image"]
        just_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(self.fake_dir, just_name+'.TIF')

        image = imread(image_path)[:, :, self.selected_channels]

        image = self.fancy_pca(image)

        image = np.transpose(image, (2, 0, 1))

        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        image = image.float()

        if self.transform:
            image = self.transform(image)

        if self.return_label:
            label = self.metadata.loc[idx, "label"]
            label = label_map_all[label]
            label = torch.tensor(label).long()
            return image.float(), label, image_name
        return image.float(), image_name

    def fancy_pca(self, image, alpha_std=0.1):
        '''
        from: https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py

        INPUTS:
        image:  numpy array with (h, w, rgb) shape, as ints between 0-255)
        alpha_std:  how much to perturb/scale the eigen vecs and vals
                    the paper used std=0.1
        RETURNS:
        numpy array as uints between 0-255

        NOTE: Depending on what is originating the image data and what is receiving
        the image data returning the values in the expected form is very important
        in having this work correctly. If you receive the image values as UINT 0-255
        then it's probably best to return in the same format. (As this
        implementation does). If the image comes in as float values ranging from
        0.0 to 1.0 then this function should be modified to return the same.
        Otherwise this can lead to very frustrating and difficult to troubleshoot
        problems in the image processing pipeline.
        This is 'Fancy PCA' from:
        # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        #######################
        #### FROM THE PAPER ###
        #######################
        "The second form of data augmentation consists of altering the intensities
        of the RGB channels in training images. Specifically, we perform PCA on the
        set of RGB pixel values throughout the ImageNet training set. To each
        training image, we add multiples of the found principal components, with
        magnitudes proportional to the corresponding eigenvalues times a random
        variable drawn from a Gaussian with mean zero and standard deviation 0.1.
        Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
        we add the following quantity:
        [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
        Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
        matrix of RGB pixel values, respectively, and αi is the aforementioned
        random variable. Each αi is drawn only once for all the pixels of a
        particular training image until that image is used for training again, at
        which point it is re-drawn. This scheme approximately captures an important
        property of natural images, namely, that object identity is invariant to
        change."
        ### END ###############
        Other useful resources for getting this working:
        # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
        # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
        '''

        orig_image = image.astype(float).copy()

        # flatten image to columns of RGB
        image_rs = image.reshape(-1, 3)

        # center mean
        image_centered = image_rs - np.mean(image_rs, axis=0)

        # paper says 3x3 covariance matrix
        image_cov = np.cov(image_centered, rowvar=False)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(image_cov)

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]


        # get [p1, p2, p3]
        m1 = np.column_stack((eig_vecs))

        # get 3x1 matrix of eigen values multiplied by random variable draw from normal
        # distribution with mean of 0 and standard deviation of alpha_std
        m2 = np.zeros((3, 1))

        # according to the paper alpha should only be draw once per augmentation (not once per channel)
        alpha = np.random.normal(0, alpha_std)

        # broad cast to speed things up
        m2[:, 0] = alpha * eig_vals[:]

        # this is the vector that we're going to add to each pixel in a moment
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):   # RGB
            orig_image[..., idx] += add_vect[idx]


        # orig_image *= 255

        # orig_image = np.clip(orig_image, 0.0, 255.0)

        # orig_image = orig_image.astype(np.uint8)

        return orig_image


def get_train_val_test(csv_file, batch_size=32, num_workers=4, fake_dir='Datasets/MAT_ACE_AS_WBC'):
    resize = 224  # image pixel size

    random_crop_scale = (0.8, 1.0)
    random_crop_ratio = (0.8, 1.2)

    # mean = [0.485, 0.456, 0.406]  # values from imagenet
    # std = [0.229, 0.224, 0.225]  # values from imagenet

    # normalization = torchvision.transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        # normalization,
        # transforms.RandomResizedCrop(resize, scale=random_crop_scale, ratio=random_crop_ratio),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    # val_transform = transforms.Compose([
    #     # normalization,
    #     transforms.Resize(resize)])
    #
    # test_transform = transforms.Compose([
    #     # normalization,
    #     transforms.Resize(resize)])

    df = pd.read_csv(csv_file, index_col=None)
    # filter out the DEV VALID WBC1 dataset
    df = df[df.dataset != 'WBC1']

    # split the dataset, getting equal percentages of labels and equal percentages of Mat_19 and Ace_20 datasets
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['label', 'dataset']])
    train, val = train_test_split(train, test_size=len(test), random_state=42, stratify=train[['label', 'dataset']])

    train_dataset = DatasetGenerator(train, transform=train_transform, fake_dir=fake_dir)
    val_dataset = DatasetGenerator(val, transform=None, fake_dir=fake_dir)
    test_dataset = DatasetGenerator(test, transform=None, fake_dir=fake_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def get_train_val(csv_file, batch_size=32, num_workers=4, fake_dir='Datasets/MAT_ACE_AS_WBC'):
    resize = 224  # image pixel size

    random_crop_scale = (0.8, 1.0)
    random_crop_ratio = (0.8, 1.2)

    # mean = [0.485, 0.456, 0.406]  # values from imagenet
    # std = [0.229, 0.224, 0.225]  # values from imagenet

    # normalization = torchvision.transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        # normalization,
        # transforms.RandomResizedCrop(resize, scale=random_crop_scale, ratio=random_crop_ratio),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    # val_transform = transforms.Compose([
    #     # normalization,
    #     transforms.Resize(resize)])
    #
    # test_transform = transforms.Compose([
    #     # normalization,
    #     transforms.Resize(resize)])

    df = pd.read_csv(csv_file, index_col=None)
    # filter out the DEV VALID WBC1 dataset
    df = df[df.dataset != 'WBC1']

    # split the dataset, getting equal percentages of labels and equal percentages of Mat_19 and Ace_20 datasets
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['label', 'dataset']])
    train, val = train_test_split(train, test_size=len(test), random_state=42, stratify=train[['label', 'dataset']])

    train = pd.concat([train, test], 0)

    train_dataset = DatasetGenerator(train, transform=train_transform, fake_dir=fake_dir)
    val_dataset = DatasetGenerator(val, transform=None, fake_dir=fake_dir)
    # test_dataset = DatasetGenerator(test, transform=None, fake_dir=fake_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader #, test_loader


class DatasetGeneratorOriginal(Dataset):

    def __init__(self,
                 metadata,
                 transform=None,
                 selected_channels=[0, 1, 2],
                 return_label=True
                 ):

        self.metadata = metadata.copy().reset_index(drop=True)
        self.transform = transform
        self.selected_channels = selected_channels
        self.return_label = return_label

    def __len__(self):
        return len(self.metadata)

    def fancy_pca(self, image, alpha_std=0.1):
        '''
        from: https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py

        INPUTS:
        image:  numpy array with (h, w, rgb) shape, as ints between 0-255)
        alpha_std:  how much to perturb/scale the eigen vecs and vals
                    the paper used std=0.1
        RETURNS:
        numpy array as uints between 0-255

        NOTE: Depending on what is originating the image data and what is receiving
        the image data returning the values in the expected form is very important
        in having this work correctly. If you receive the image values as UINT 0-255
        then it's probably best to return in the same format. (As this
        implementation does). If the image comes in as float values ranging from
        0.0 to 1.0 then this function should be modified to return the same.
        Otherwise this can lead to very frustrating and difficult to troubleshoot
        problems in the image processing pipeline.
        This is 'Fancy PCA' from:
        # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        #######################
        #### FROM THE PAPER ###
        #######################
        "The second form of data augmentation consists of altering the intensities
        of the RGB channels in training images. Specifically, we perform PCA on the
        set of RGB pixel values throughout the ImageNet training set. To each
        training image, we add multiples of the found principal components, with
        magnitudes proportional to the corresponding eigenvalues times a random
        variable drawn from a Gaussian with mean zero and standard deviation 0.1.
        Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
        we add the following quantity:
        [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
        Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
        matrix of RGB pixel values, respectively, and αi is the aforementioned
        random variable. Each αi is drawn only once for all the pixels of a
        particular training image until that image is used for training again, at
        which point it is re-drawn. This scheme approximately captures an important
        property of natural images, namely, that object identity is invariant to
        change."
        ### END ###############
        Other useful resources for getting this working:
        # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
        # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
        '''

        orig_image = image.astype(float).copy()

        # flatten image to columns of RGB
        image_rs = image.reshape(-1, 3)

        # center mean
        image_centered = image_rs - np.mean(image_rs, axis=0)

        # paper says 3x3 covariance matrix
        image_cov = np.cov(image_centered, rowvar=False)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(image_cov)

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]


        # get [p1, p2, p3]
        m1 = np.column_stack((eig_vecs))

        # get 3x1 matrix of eigen values multiplied by random variable draw from normal
        # distribution with mean of 0 and standard deviation of alpha_std
        m2 = np.zeros((3, 1))

        # according to the paper alpha should only be draw once per augmentation (not once per channel)
        alpha = np.random.normal(0, alpha_std)

        # broad cast to speed things up
        m2[:, 0] = alpha * eig_vals[:]

        # this is the vector that we're going to add to each pixel in a moment
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):   # RGB
            orig_image[..., idx] += add_vect[idx]


        # orig_image *= 255

        # orig_image = np.clip(orig_image, 0.0, 255.0)

        # orig_image = orig_image.astype(np.uint8)

        return orig_image

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset = self.metadata.loc[idx, "dataset"]
        image_name = self.metadata.loc[idx, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.metadata.loc[idx, "file"]
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
        image = image.float()

        if self.transform:
            image = self.transform(image)

        if self.return_label:
            label = self.metadata.loc[idx, "label"]
            label = label_map_all[label]
            label = torch.tensor(label).long()
            return image.float(), label, image_name
        return image.float(), image_name


def get_pred_loader(csv_file, batch_size=1, num_workers=4):
    resize = 224  # image pixel size

    # mean = [0.485, 0.456, 0.406]  # values from imagenet
    # std = [0.229, 0.224, 0.225]  # values from imagenet

    # normalization = torchvision.transforms.Normalize(mean, std)

    pred_transform = transforms.Compose([
        # normalization,
        transforms.Resize(resize)])

    df = pd.read_csv(csv_file, index_col=None)
    # get the DEV VALID WBC1 dataset
    df = df[df.dataset == 'WBC1']

    pred_dataset = DatasetGeneratorOriginal(df, transform=pred_transform, return_label=False)

    pred_loader = DataLoader(
        pred_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return pred_loader


class DatasetGeneratorOriginalMeanStd(Dataset):

    def __init__(self,
                 metadata,
                 transform=None,
                 selected_channels=[0, 1, 2],
                 return_label=True
                 ):

        self.metadata = metadata.copy().reset_index(drop=True)
        self.transform = transform
        self.selected_channels = selected_channels
        self.return_label = return_label

    def __len__(self):
        return len(self.metadata)

    def fancy_pca(self, image, alpha_std=0.1):
        '''
        from: https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py

        INPUTS:
        image:  numpy array with (h, w, rgb) shape, as ints between 0-255)
        alpha_std:  how much to perturb/scale the eigen vecs and vals
                    the paper used std=0.1
        RETURNS:
        numpy array as uints between 0-255

        NOTE: Depending on what is originating the image data and what is receiving
        the image data returning the values in the expected form is very important
        in having this work correctly. If you receive the image values as UINT 0-255
        then it's probably best to return in the same format. (As this
        implementation does). If the image comes in as float values ranging from
        0.0 to 1.0 then this function should be modified to return the same.
        Otherwise this can lead to very frustrating and difficult to troubleshoot
        problems in the image processing pipeline.
        This is 'Fancy PCA' from:
        # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        #######################
        #### FROM THE PAPER ###
        #######################
        "The second form of data augmentation consists of altering the intensities
        of the RGB channels in training images. Specifically, we perform PCA on the
        set of RGB pixel values throughout the ImageNet training set. To each
        training image, we add multiples of the found principal components, with
        magnitudes proportional to the corresponding eigenvalues times a random
        variable drawn from a Gaussian with mean zero and standard deviation 0.1.
        Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
        we add the following quantity:
        [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
        Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
        matrix of RGB pixel values, respectively, and αi is the aforementioned
        random variable. Each αi is drawn only once for all the pixels of a
        particular training image until that image is used for training again, at
        which point it is re-drawn. This scheme approximately captures an important
        property of natural images, namely, that object identity is invariant to
        change."
        ### END ###############
        Other useful resources for getting this working:
        # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
        # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
        '''

        image = image/255
        orig_image = image.astype(float).copy() #range 0 - 255


        # flatten image to columns of RGB
        image_rs = image.reshape(-1, 3)


        # center mean
        image_centered = image_rs - np.mean(image_rs, axis=0)

        # paper says 3x3 covariance matrix
        image_cov = np.cov(image_centered, rowvar=False)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(image_cov)

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]


        # get [p1, p2, p3]
        m1 = np.column_stack((eig_vecs))

        # get 3x1 matrix of eigen values multiplied by random variable draw from normal
        # distribution with mean of 0 and standard deviation of alpha_std
        m2 = np.zeros((3, 1))

        # according to the paper alpha should only be draw once per augmentation (not once per channel)
        alpha = np.random.normal(0, alpha_std)

        # broad cast to speed things up
        m2[:, 0] = alpha * eig_vals[:]

        # this is the vector that we're going to add to each pixel in a moment
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):   # RGB
            orig_image[..., idx] += add_vect[idx]


        orig_image *= 255

        orig_image = np.clip(orig_image, 0.0, 255.0)

        orig_image = orig_image.astype(np.uint8)


        return orig_image

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset = self.metadata.loc[idx, "dataset"]
        image_name = self.metadata.loc[idx, "Image"]
        crop_size = dataset_image_size[dataset]

        h5_file_path = self.metadata.loc[idx, "file"]
        image = imread(h5_file_path)[:, :, self.selected_channels]

        image = self.fancy_pca(image)

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
        image = image.float()

        if self.transform:
            image = self.transform(image)

        if self.return_label:
            label = self.metadata.loc[idx, "label"]
            label = label_map_all[label]
            label = torch.tensor(label).long()
            return image.float(), label, image_name
        return image.float(), image_name


class DatasetGeneratorTestPhaseMeanStd(Dataset):

    def __init__(self,
                 image_paths,
                 transform=None,
                 selected_channels=[0, 1, 2],
                 return_label=True
                 ):

        self.image_paths = image_paths
        self.transform = transform
        self.selected_channels = selected_channels
        self.return_label = return_label

    def __len__(self):
        return len(self.image_paths)

    def fancy_pca(self, image, alpha_std=0.1):
        '''
        from: https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py

        INPUTS:
        image:  numpy array with (h, w, rgb) shape, as ints between 0-255)
        alpha_std:  how much to perturb/scale the eigen vecs and vals
                    the paper used std=0.1
        RETURNS:
        numpy array as uints between 0-255

        NOTE: Depending on what is originating the image data and what is receiving
        the image data returning the values in the expected form is very important
        in having this work correctly. If you receive the image values as UINT 0-255
        then it's probably best to return in the same format. (As this
        implementation does). If the image comes in as float values ranging from
        0.0 to 1.0 then this function should be modified to return the same.
        Otherwise this can lead to very frustrating and difficult to troubleshoot
        problems in the image processing pipeline.
        This is 'Fancy PCA' from:
        # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        #######################
        #### FROM THE PAPER ###
        #######################
        "The second form of data augmentation consists of altering the intensities
        of the RGB channels in training images. Specifically, we perform PCA on the
        set of RGB pixel values throughout the ImageNet training set. To each
        training image, we add multiples of the found principal components, with
        magnitudes proportional to the corresponding eigenvalues times a random
        variable drawn from a Gaussian with mean zero and standard deviation 0.1.
        Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
        we add the following quantity:
        [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
        Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
        matrix of RGB pixel values, respectively, and αi is the aforementioned
        random variable. Each αi is drawn only once for all the pixels of a
        particular training image until that image is used for training again, at
        which point it is re-drawn. This scheme approximately captures an important
        property of natural images, namely, that object identity is invariant to
        change."
        ### END ###############
        Other useful resources for getting this working:
        # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
        # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
        '''

        image = image/255
        orig_image = image.astype(float).copy() #range 0 - 255


        # flatten image to columns of RGB
        image_rs = image.reshape(-1, 3)


        # center mean
        image_centered = image_rs - np.mean(image_rs, axis=0)

        # paper says 3x3 covariance matrix
        image_cov = np.cov(image_centered, rowvar=False)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(image_cov)

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]


        # get [p1, p2, p3]
        m1 = np.column_stack((eig_vecs))

        # get 3x1 matrix of eigen values multiplied by random variable draw from normal
        # distribution with mean of 0 and standard deviation of alpha_std
        m2 = np.zeros((3, 1))

        # according to the paper alpha should only be draw once per augmentation (not once per channel)
        alpha = np.random.normal(0, alpha_std)

        # broad cast to speed things up
        m2[:, 0] = alpha * eig_vals[:]

        # this is the vector that we're going to add to each pixel in a moment
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):   # RGB
            orig_image[..., idx] += add_vect[idx]


        orig_image *= 255

        orig_image = np.clip(orig_image, 0.0, 255.0)

        orig_image = orig_image.astype(np.uint8)


        return orig_image

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        crop_size = crop_WBC1

        h5_file_path = self.image_paths[idx]  # self.metadata.loc[idx, "file"]
        image_name = os.path.basename(h5_file_path)
        image = imread(h5_file_path)[:, :, self.selected_channels]

        image = self.fancy_pca(image)

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
        image = image.float()

        if self.transform:
            image = self.transform(image)

        return image.float(), image_name


def get_pred_loader_mean_std(csv_file, batch_size=1, num_workers=4):
    resize = 224  # image pixel size

    with open('Datasets/Mean_image.pickle', 'rb') as f:
        mean_im = pickle.load(f)
    with open('Datasets/Std_image.pickle', 'rb') as f:
        std_im = pickle.load(f)

    ####
    normalization = transforms.Lambda(lambda im: (im - mean_im) / std_im)

    pred_transform = transforms.Compose([
        transforms.Resize(resize),
        normalization])

    df = pd.read_csv(csv_file, index_col=None)
    # get the DEV VALID WBC1 dataset
    df = df[df.dataset == 'WBC1']

    pred_dataset = DatasetGeneratorOriginalMeanStd(df, transform=pred_transform, return_label=False)

    pred_loader = DataLoader(
        pred_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return pred_loader


def get_test_phase_data(test_dir, batch_size=1, num_workers=4):
    resize = 224  # image pixel size

    with open('Datasets/Mean_image.pickle', 'rb') as f:
        mean_im = pickle.load(f)
    with open('Datasets/Std_image.pickle', 'rb') as f:
        std_im = pickle.load(f)

    ####
    normalization = transforms.Lambda(lambda im: (im - mean_im) / std_im)

    pred_transform = transforms.Compose([
        transforms.Resize(resize),
        normalization])

    image_paths = glob(os.path.join(test_dir, '*'))

    pred_dataset = DatasetGeneratorTestPhaseMeanStd(image_paths, transform=pred_transform, return_label=False)

    pred_loader = DataLoader(
        pred_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return pred_loader











