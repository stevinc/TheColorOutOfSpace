import argparse
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch import nn
from torch.utils.data import SubsetRandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from Colorization.dataset.dataset_big_earth import Color
import albumentations as A
from matplotlib import pyplot as plt


class BigEarthDatasetTorch(Dataset):
    def __init__(self, csv_path: str, random_seed: int, bands_indices: list, img_size: int,
                 augmentation: int, n_samples=100000):
        """
        Args:
            csv_path: path to csv file containing paths to images
            quantiles: path to json file containing quantiles of each bands
            random_seed: seed value
            bands: list of the bands to consider for the training
            n_samples: number of samples to exploit for the training
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Load dataset
        self.folder_path, self.labels_name, self.labels_class = self.load_dataset(csv_path, random_seed, n_samples)
        # Calculate len
        self.data_len = len(self.folder_path)
        print("Dataset len: ", self.data_len)
        # image size
        self.img_size = img_size
        # bands
        self.bands_indices = torch.BoolTensor(bands_indices)
        # augmentation
        self.augmentation = augmentation

    @staticmethod
    def load_dataset(csv_path: str, random_seed: int, n_samples: int) -> Tuple[list, list, list]:
        """
        function to load the dataset from the csv path
        :param csv_path: path to the csv file
        :param random_seed: seed
        :param n_samples: n_samples to considerer for the training
        :return: list of paths to the images
        """
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None)
        # First column contains the folder paths
        folder_path = data_info.iloc[:n_samples, 0].tolist()
        # Second column contains the text labels
        labels_name = data_info.iloc[:n_samples, 1].tolist()
        # Third column contains the number labels
        labels_class = data_info.iloc[:n_samples, 2].tolist()
        # shuffle the entries, specify the seed
        tmp_shuffle = list(zip(folder_path, labels_name, labels_class))
        np.random.seed(random_seed)
        np.random.shuffle(tmp_shuffle)
        folder_path, labels_name, labels_class = zip(*tmp_shuffle)
        # for the colorization version return only the image paths
        return folder_path, labels_name, labels_class

    def split_dataset(self, thresh_test: float, thresh_val: float) -> Union[Tuple[list, list, list], Tuple[list, list]]:
        """
        :param thresh_test: threshold for splitting the dataset in training and test set
        :param thresh_val: threshold for splitting the dataset in training, test and val set
        :return: the two split (or three, if I add the validation set)
        """
        indices = list(range(self.data_len))
        split_test = int(np.floor(thresh_test * self.data_len))
        if thresh_val is not None:
            split_val = int(np.floor(thresh_val * self.data_len))
            return indices[split_val:], indices[split_test:split_val], indices[:split_test]
        else:
            return indices[split_test:], indices[:split_test]

    @staticmethod
    def split_bands(spectral_img: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        function to split the bands between rgb and other bands
        :param spectral_img: complete image
        :param bands_indices: indices of the spectral bands to keep
        :return: splitted image
        """
        indices = torch.tensor([0, 4, 5, 6, 7, 8, 9, 10, 11])
        indices_rgb = torch.tensor([3, 2, 1])
        rgb = torch.index_select(input=spectral_img, dim=0, index=indices_rgb)
        spectral_bands = torch.index_select(input=spectral_img, dim=0, index=indices)
        return spectral_bands, rgb

    @staticmethod
    def augmentation_fn(images: torch.Tensor) -> torch.Tensor:
        """
        function that applies data augmentation to torch image
        :param images: current image
        :return: augmented image
        """
        rnd = np.random.random_sample()
        images = torch.unsqueeze(images, dim=1)
        angle = np.random.randint(-15, 15)

        for id, image in enumerate(images):
            if rnd < 0.25:
                image = TF.to_pil_image(image)
                image = TF.rotate(image, angle)
            elif 0.25 <= rnd <= 0.50:
                image = TF.to_pil_image(image)
                image = TF.vflip(image)
            elif 0.50 < rnd <= 0.75:
                image = TF.to_pil_image(image)
                image = TF.hflip(image)
            else:
                images = torch.squeeze(images)
                return images
            images[id] = TF.to_tensor(image)
        return torch.squeeze(images)

    @staticmethod
    def album_aug(images: torch.Tensor) -> torch.Tensor:
        """
        function that applies data augmentation using the albumentations library to speed up
        :param images: current image
        :return: augmented image
        """
        angle = np.random.randint(-15, 15)
        transform = A.Compose([
            A.OneOf([
                A.Rotate(limit=angle, always_apply=False, p=0.33),
                A.VerticalFlip(p=0.33),
                A.HorizontalFlip(p=0.33)
            ],
                p=0.75)]
        )
        image_aug = np.transpose(images.numpy(), (1, 2, 0))
        image_aug = transform(image=image_aug)['image']
        return torch.squeeze(torch.from_numpy(image_aug))

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # obtain the right folder
        imgs_file = self.folder_path[index]
        # load torch image
        spectral_img = torch.load(imgs_file + '/all_bands_chroma.pt')
        # resize the image as specified in the params dsize
        spectral_img = torch.squeeze(
            nn.functional.interpolate(input=torch.unsqueeze(spectral_img, dim=0), size=self.img_size))
        # take only the bands specified in the init
        spectral_img = spectral_img[self.bands_indices]
        # eventually apply augmentation
        if self.augmentation:
            spectral_img = self.augmentation_fn(spectral_img)
        # split the bands and convert to CieLab space
        spectral_bands, rgb = self.split_bands(spectral_img)
        # convert tensor to numpy
        rgb = np.transpose(rgb.numpy(), (1, 2, 0))
        L, ab = Color.rgb2lab(rgb)
        return spectral_bands, self.to_tensor(L), self.to_tensor(ab)

    def __len__(self) -> int:
        return self.data_len


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='BigEarthNet dataset tiff version')
    argparser.add_argument('--csv_filename', type=str,
                           default='BigEarthNet_all_refactored_no_clouds_and_snow_v2_new_path.csv',
                           required=True, help='csv containing dataset paths')
    argparser.add_argument('--n_samples', type=int, default=3000, help='Number of samples to create the csv file')
    argparser.add_argument('--augmentation', type=int, default=1, choices=[0, 1], help='set to 1 for use augmenation')

    args = argparser.parse_args()
    # Dataset definition
    big_earth = BigEarthDatasetTorch(csv_path=args.csv_filename, random_seed=19,
                                     bands_indices=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], img_size=128,
                                     augmentation=1, n_samples=args.n_samples)
    # dataset split
    train_idx, val_idx, test_idx = big_earth.split_dataset(0.2, 0.4)
    # dataset sampler
    train_sampler = SequentialSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # dataset loader
    train_loader = torch.utils.data.DataLoader(big_earth, batch_size=16,
                                               sampler=train_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(big_earth, batch_size=1,
                                              sampler=test_sampler, num_workers=4)
    start_time = time.time()

    runs = 5
    for i in range(runs):
        for idx, (spectral_img, L, ab) in enumerate(train_loader):
            print(idx)

    print("Mean Time over 5 runs: ", (time.time() - start_time) / runs)
