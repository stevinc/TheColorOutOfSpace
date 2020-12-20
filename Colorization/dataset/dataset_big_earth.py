import argparse
import glob
import json
import time
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms


# "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
def torch2numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    function to translate torch tensor in np array, remove batch size 1
    :param tensor: input torch tensor
    :return: ndarray
    """
    tensor = torch.squeeze(tensor, dim=0)
    return np.transpose(tensor.numpy(), (1, 2, 0))


def load_dict_from_json(json_path: str) -> dict:
    """
    function to load json
    :param json_path: path to the file
    :return: dict
    """
    with open(json_path) as f:
        params = json.load(f)
        return params


class Color:
    def __init__(self):
        pass

    @staticmethod
    def rgb2gray(rgb: np.ndarray) -> np.ndarray:
        """
        function which convert an RGB image to grayscale
        :param rgb: input image
        :return: grayscale image
        """
        return np.reshape(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]),
                          (rgb.shape[0], rgb.shape[1], 1)).astype(np.float32)

    @staticmethod
    def rgb2lab(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        function which convert an rgb image to lab, normalized between [0-1]
        :param rgb: input image
        :return: converted image
        """
        lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab_img[:, :, 0] *= 255 / 100
        lab_img[:, :, 1] += 128
        lab_img[:, :, 2] += 128
        lab_img /= 255
        return lab_img[:, :, 0], lab_img[:, :, 1:]

    @staticmethod
    def lab2rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
        """
       function which convert a lab image to rgb, normalized between [0-1]
       :param L: input channel
       :param ab: input channels
       :return: converted image
       """
        L = L.cpu().numpy()
        ab = ab.cpu().detach().numpy()
        Lab = np.concatenate((L, ab), axis=1)
        Lab = np.transpose(Lab, (0, 2, 3, 1))
        B, W, H, C = Lab.shape[0], Lab.shape[1], Lab.shape[2], Lab.shape[3]
        # reshape to convert all the images in the batch without iteration
        Lab = np.reshape(Lab, (B * W, H, C))
        Lab *= 255
        Lab[:, :, 0] *= 100 / 255
        Lab[:, :, 1] -= 128
        Lab[:, :, 2] -= 128
        rgb = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB)
        rgb = np.reshape(rgb, (B, W, H, C))
        rgb = np.transpose(rgb, (0, 3, 1, 2))
        rgb = torch.from_numpy(rgb)
        return rgb


class BigEarthDataset(Dataset):
    def __init__(self, csv_path: str, quantiles: str, random_seed: int, bands: list,
                 create_torch_dataset=0, n_samples=100000):
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
        # load quantiles json file
        self.quantiles = load_dict_from_json(quantiles)
        # bands
        self.bands = bands
        # flag for create dataset torch version
        self.create_torch_dataset = create_torch_dataset

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

    def quantiles_std(self, img: np.ndarray, band: list, quantiles: dict) -> np.ndarray:
        """
        function that normalize the input bands to [0-1]
        :param img: input image
        :param band: list of bands
        :param quantiles: dict containing the min-max quantiles for each band
        :return: normalized image
        """
        min_q = quantiles[band]['min_q']
        max_q = quantiles[band]['max_q']
        img[img < min_q] = min_q
        img[img > max_q] = max_q
        img_dest = np.zeros_like(img)
        img_dest = cv2.normalize(img, img_dest, 0, 255, cv2.NORM_MINMAX)
        img_dest = img_dest.astype(np.float32) / 255.
        return img_dest

    def custom_loader(self, path: str, band: list, quantiles: dict) -> np.ndarray:
        """
        function to open the image
        :param path: path to the image to load
        :param band: list of bands
        :param quantiles: dict containing the min-max quantiles for each band
        :return: ndarray image resized
        """
        # read the band as it is, with IMREAD_UNCHANGED
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        img = self.quantiles_std(img, band, quantiles)
        w, h = img.shape
        return img.reshape(w, h, 1)

    # split the bands between rgb and all the others
    def split_bands(self, spectral_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        function to split the bands between rgb and other bands
        :param spectral_img: complete image
        :return: splitted image
        """
        indices = [0, 4, 5, 6, 7, 8, 9, 10, 11]
        indices_rgb = [3, 2, 1]
        spectral_bands = np.take(spectral_img, indices=indices, axis=2)
        rgb = np.take(spectral_img, indices=indices_rgb, axis=2)
        return spectral_bands, rgb

    def save_torch_dataset(self, imgs_file: str, spectral_img: np.ndarray) -> True:
        """
        function to save a torch version of the dataset
        :param imgs_file: path to the current image
        :param spectral_img: current image
        :return: true
        """
        parts = list(Path(imgs_file).parts)
        parts[4] = 'BigEarthNet_torch_version_v2'
        new_path = Path(*parts)
        new_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_tensor(spectral_img), new_path / 'all_bands.pt')
        return True

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # obtain the right folder
        imgs_file = self.folder_path[index] # [2:-2] # to remove [\ \]
        imgs_bands = []
        for b in self.bands:
            for filename in glob.iglob(imgs_file+"/*" + b + ".tif"):
                band = self.custom_loader(filename, b, self.quantiles)
                imgs_bands.append(band)
        spectral_img = np.concatenate(imgs_bands, axis=2)
        if self.create_torch_dataset:
            self.save_torch_dataset(imgs_file, spectral_img)
        spectral_bands, rgb = self.split_bands(spectral_img)
        L, ab = Color.rgb2lab(rgb)
        return self.to_tensor(spectral_bands), self.to_tensor(L), self.to_tensor(ab)

    def __len__(self) -> int:
        return self.data_len


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='BigEarthNet dataset tiff version')
    argparser.add_argument('--csv_filename', type=str, default='BigEarthNet_all_refactored_no_clouds_and_snow_server.csv',
                           required=True, help='csv containing dataset paths')
    argparser.add_argument('--n_samples', type=int, default=3000, help='Number of samples to create the csv file')
    argparser.add_argument('--create_torch_dataset', type=int, default=0, choices=[0, 1], help='set 1 to create torch'
                                                                                               'version and 0 to use the '
                                                                                               'training dataset')
    args = argparser.parse_args()
    # Dataset definition
    big_earth = BigEarthDataset(csv_path=args.csv_filename, quantiles='quantiles_3000.json',
                                random_seed=19, bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
                                create_torch_dataset=args.create_torch_dataset, n_samples=args.n_samples)
    # dataset split
    train_idx, val_idx, test_idx = big_earth.split_dataset(0.2, 0.4)
    # dataset sampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # dataset loader
    train_loader = torch.utils.data.DataLoader(big_earth, batch_size=16,
                                               sampler=train_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(big_earth, batch_size=16,
                                              sampler=test_sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(big_earth, batch_size=16,
                                              sampler=val_sampler, num_workers=4)
    start_time = time.time()

    for idx, (spectral_img, L, ab) in enumerate(train_loader):
        print(idx)

    for idx, (spectral_img, L, ab) in enumerate(test_loader):
        print(idx)

    for idx, (spectral_img, L, ab) in enumerate(val_loader):
        print(idx)

    print("time: ", time.time() - start_time)