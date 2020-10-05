import argparse
import glob
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler

from Colorization.dataset.dataset_big_earth_torch import BigEarthDatasetTorch


class BigEarthDatasetTorchMLC(BigEarthDatasetTorch):
    def __init__(self, csv_path: str, random_seed: int, bands_indices: list, img_size: int, n_samples=100000):
        BigEarthDatasetTorch.__init__(self, csv_path=csv_path, random_seed=random_seed, bands_indices=bands_indices,
                                      img_size=img_size, augmentation=0, n_samples=n_samples)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        # obtain the right folder
        imgs_file = self.folder_path[index]
        # load torch image
        spectral_img = torch.load(imgs_file + '/all_bands_chroma.pt')
        # resize the image as specified in the params dsize
        spectral_img = torch.squeeze(
            nn.functional.interpolate(input=torch.unsqueeze(spectral_img, dim=0), size=self.img_size))
        # take only the bands specified in the init
        spectral_img = spectral_img[self.bands_indices]
        # if RGB: invert the indices as it is saved as BGR
        if sum(self.bands_indices) == 3:
            spectral_img = torch.flip(spectral_img, [0])
        # create multi-hot labels vector
        labels_index = list(map(int, self.labels_class[index][1:-1].split(',')))
        labels_class = np.zeros(19)
        labels_class[labels_index] = 1
        return spectral_img, torch.tensor(labels_class)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='BigEarthNetTorchMLC dataset version')
    argparser.add_argument('--csv_filename', type=str,
                           default='BigEarthNet_all_refactored_no_clouds_and_snow_server.csv',
                           required=True, help='csv containing dataset paths')
    argparser.add_argument('--n_samples', type=int, default=3000, help='Number of samples to create the csv file')

    args = argparser.parse_args()
    # Dataset definition
    big_earth = BigEarthDatasetTorchMLC(csv_path=args.csv_filename, random_seed=19, bands_indices=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        img_size=128, n_samples=args.n_samples)
    # dataset split
    train_idx, val_idx, test_idx = big_earth.split_dataset(0.2, 0.4)
    # dataset sampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # dataset loader
    train_loader = torch.utils.data.DataLoader(big_earth, batch_size=16,
                                               sampler=train_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(big_earth, batch_size=1,
                                              sampler=test_sampler, num_workers=0)
    start_time = time.time()

    for idx, (spectral_img, labels) in enumerate(train_loader):
        print(idx)

    print("time: ", time.time() - start_time)
