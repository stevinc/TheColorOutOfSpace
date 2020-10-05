import argparse
import csv
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

Land_cover_Classes = {
    'Mixed forest': 0,
    'Coniferous forest': 1,
    'Non-irrigated arable land': 2,
    'Transitional woodland/shrub': 3,
    'Broad-leaved forest': 4,
    'Land principally occupied by agriculture, with significant areas of natural vegetation': 5,
    'Complex cultivation patterns': 6,
    'Pastures': 7,
    'Water bodies': 8,
    'Sea and ocean': 9,
    'Discontinuous urban fabric':10,
    'Agro-forestry areas': 11,
    'Peatbogs': 12,
    'Permanently irrigated land': 13,
    'Industrial or commercial units': 14,
    'Natural grassland': 15,
    'Olive groves': 16,
    'Sclerophyllous vegetation': 17,
    'Continuous urban fabric': 18,
    'Water courses': 19,
    'Vineyards': 20,
    'Annual crops associated with permanent crops': 21,
    'Inland marshes': 22,
    'Moors and heathland': 23,
    'Sport and leisure facilities': 24,
    'Fruit trees and berry plantations': 25,
    'Mineral extraction sites': 26,
    'Rice fields': 27,
    'Road and rail networks and associated land': 28,
    'Bare rock': 29,
    'Green urban areas': 30,
    'Beaches, dunes, sands': 31,
    'Sparsely vegetated areas': 32,
    'Salt marshes': 33,
    'Coastal lagoons': 34,
    'Construction sites': 35,
    'Estuaries': 36,
    'Intertidal flats': 37,
    'Airports': 38,
    'Dump sites': 39,
    'Port areas': 40,
    'Salines': 41,
    'Burnt areas': 42
}


class BigEarthUtils:
    def __init__(self):
        pass

    @staticmethod
    def big_earth_to_csv(big_e_path: str, num_samples: int, csv_filename: str) -> True:
        """
        Function which generate the csv file of all or a portion of the BigEarth dataset
        :param big_e_path: path to BigEarth dataset
        :param num_samples: number of samples to consider in the creation of the csv file (-1 to select all dataset)
        :param csv_filename: name of the created file
        :return: True
        """
        path = Path(big_e_path)
        print("collecting dirs...")
        start_time = time.time()
        labels_names = []
        labels_values = []
        if num_samples == -1:
            dirs = [str(e) for e in path.iterdir() if e.is_dir()]
        else:
            # zip and range() to choose only a specific number of example
            dirs = [str(e) for _, e in zip(range(num_samples), path.iterdir()) if e.is_dir()]
        for idx, d in enumerate(dirs):
            for e in glob.glob(d + "/*.json"):
                with open(e) as f:
                    j_file = json.load(f)
                    labels_names.append(j_file['labels'])
                    labels_values.append([Land_cover_Classes[label] for label in j_file['labels']])
        # write the dirs on a csv file
        print("writing on csv...")
        things_to_write = zip(dirs, labels_names, labels_values)
        with open(csv_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(things_to_write)
        print(f"finishing in : {time.time() - start_time}")
        return True

    @staticmethod
    def min_max_quantile(csv_filename: str, n_samples: int) -> dict:
        """
        Function that compute the
        :param csv_filename: path of the csv_filename of the BigEarth dataset
        :param n_samples: number of samples to use for calculate the min and max quantile
        :return: a dict containing min and max quantiles for every sentinel-2 bands
        """
        bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        data = pd.read_csv(csv_filename, header=None)
        paths = data.iloc[:, 0].tolist()
        quantiles = {}
        for b in bands:
            imgs = []
            for i in range(n_samples):
                path = paths[i]  # i choose the i-th path of the list
                for filename in glob.iglob(path + "/*" + b + ".tif"):
                    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                    imgs.append(img)
            imgs = np.stack(imgs, axis=0).reshape(-1)
            quantiles[b] = {
                'min_q': np.quantile(imgs, 0.02),
                'max_q': np.quantile(imgs, 0.98)
            }
            print(b, quantiles[b])
        return quantiles

    @staticmethod
    def save_dict_to_json(d: dict, json_path: str) -> None:
        with open(json_path, 'w') as f:
            json.dump(d, f, indent=4)

    @staticmethod
    def delete_patches(csv_filename: str) -> True:
        """
        Function which delete images covered by cloud or snow
        :param csv_filename: Dataset file created above
        :return: True
        """
        csv_snow_patches = 'patches_with_seasonal_snow.csv'
        csv_clouds_patches = 'patches_with_cloud_and_shadow.csv'
        data = pd.read_csv(csv_filename, header=None)
        snow_patches = pd.read_csv(Path.cwd() / csv_snow_patches, header=None)
        clouds_patches = pd.read_csv(Path.cwd() / csv_clouds_patches, header=None)
        patches = snow_patches.iloc[:, 0].tolist() + clouds_patches.iloc[:, 0].tolist()
        df = data[~data.iloc[:, 0].str.contains('|'.join(patches))]
        df.to_csv(csv_filename[:-4] + '_no_clouds_and_snow_server' + csv_filename[-4:], header=None, index=False)
        return True

    @staticmethod
    def delete_patches_v2(csv_filename: str) -> True:
        """
       Function which delete images covered by cloud or snow
       :param csv_filename: Dataset file created above
       :return: True
       """
        data = pd.read_csv(csv_filename, header=None)
        data_copy = data.copy()
        data_copy = data_copy.replace({"/nas/softechict-nas-2/svincenzi/BigEarthNet-v1.0/": ""}, regex=True)
        csv_snow_patches = 'patches_with_seasonal_snow.csv'
        csv_clouds_patches = 'patches_with_cloud_and_shadow.csv'
        snow_patches = pd.read_csv(Path.cwd() / csv_snow_patches, header=None)
        clouds_patches = pd.read_csv(Path.cwd() / csv_clouds_patches, header=None)
        patches = snow_patches.iloc[:, 0].tolist() + clouds_patches.iloc[:, 0].tolist()
        data = data[~data_copy.iloc[:, 0].isin(patches)]
        data.to_csv(csv_filename[:-4] + '_no_clouds_and_snow_v2' + csv_filename[-4:], header=None, index=False)
        return True

    @staticmethod
    def replace_path_csv(csv_filename: str, new_path: str) -> True:
        """
        function that change the path in the csv file
        :param csv_filename: Dataset file
        :param new_path: new path to set in the csv file
        :return: True
        """
        data = pd.read_csv(csv_filename, header=None)
        data = data.replace({"/nas/softechict-nas-2/svincenzi/BigEarthNet-v1.0/": new_path}, regex=True)
        data.to_csv(csv_filename[:-4] + '_new_path' + csv_filename[-4:], header=None, index=False)
        return True


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='BigEarthNet utils')
    argparser.add_argument('--big_e_path', type=str, default=None, required=True, help='path to the BigEarth dataset')
    argparser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to create the csv file')
    argparser.add_argument('--csv_filename', type=str, default='BigEarth.csv', help='Name of the csv dataset file')
    argparser.add_argument('--n_samples', type=int, default=3000, help='Number of samples to calculate the min-max quantile')
    argparser.add_argument('--mode', default='csv_creation', choices=['csv_creation', 'delete_patches', 'delete_patches_v2',
                                                                      'quantiles', 'replace_path_csv'],
                           type=str, help='select the action to perform: csv_creation, delete_patches, '
                                          'delete_patches_v2, quantiles or replace_path_csv')
    argparser.add_argument('--new_path_csv', type=str, default=None, help='indicate the new path to change the csv')

    args = argparser.parse_args()
    # csv creation
    if args.mode == 'csv_creation':
        BigEarthUtils.big_earth_to_csv(args.big_e_path, args.num_samples, args.csv_filename)
    # delete patches
    elif args.mode == 'delete_patches':
         BigEarthUtils.delete_patches(args.csv_filename)
    # delete patches_v2
    elif args.mode == 'delete_patches_v2':
         BigEarthUtils.delete_patches_v2(args.csv_filename)
    # min-max quantiles
    elif args.mode == 'quantiles':
        quantiles = BigEarthUtils.min_max_quantile(args.csv_filename, args.n_samples)
        # save the quantiles on a json file
        BigEarthUtils.save_dict_to_json(quantiles, f"quantiles_{args.n_samples}.json")
    # replace csv path
    elif args.mode == 'replace_path_csv':
        BigEarthUtils.replace_path_csv(args.csv_filename, args.new_path_csv)


