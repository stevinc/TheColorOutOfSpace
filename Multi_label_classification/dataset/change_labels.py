import argparse
import pandas as pd

# OLD LABELS TO DELETE
labels_to_delete_index = [28, 40, 38, 26, 39, 35, 30, 24, 29, 42, 37]

labels_to_delete_name = ["Road and rail networks and associated land",
                         "Port areas",
                         "Airports",
                         "Mineral extraction sites",
                         "Dump sites",
                         "Construction sites",
                         "Green urban areas",
                         "Sport and leisure facilities",
                         "Bare rock",
                         "Burnt areas",
                         "Intertidal flats"]

# New labels assignments between old and new labels
new_labels_assignments = {
    0: 10,
    1: 9,
    2: 2,
    3: 13,
    4: 8,
    5: 6,
    6: 5,
    7: 4,
    8: 17,
    9: 18,
    10: 0,
    11: 7,
    12: 15,
    13: 2,
    14: 1,
    15: 11,
    16: 3,
    17: 12,
    18: 0,
    19: 17,
    20: 3,
    21: 3,
    22: 15,
    23: 12,
    25: 3,
    27: 2,
    31: 14,
    32: 11,
    33: 16,
    34: 18,
    36: 18,
    41: 16
}

# New Labels
BigEarthNet19_labels = {
    0: "Urban fabric",
    1: "Industrial or commercial units",
    2: "Arable land",
    3: "Permanent crops",
    4: "Pastures",
    5: "Complex cultivation patterns",
    6: "Land principally occupied by agriculture, with significant areas of natural vegetation",
    7: "Agro-forestry areas",
    8: "Broad-leaved forest",
    9: "Coniferous forest",
    10: "Mixed forest",
    11: "Natural grassland and sparsely vegetated areas",
    12: "Moors, heathland and sclerophyllous vegetation",
    13: "Transitional woodland, shrub",
    14: "Beaches, dunes, sands",
    15: "Inland wetlands",
    16: "Coastal wetlands",
    17: "Inland waters",
    18: "Marine waters"
}


class Labels:
    def __init__(self):
        pass

    @staticmethod
    def delete_labels_in_csv(csv_filename: str, new_csv_file: str) -> True:
        """
        Function that deletes the labels no more present in the new nomenclature
        :param csv_filename: old csv file
        :param new_csv_file: new csv file
        :return: True
        """
        csv_file = pd.read_csv(csv_filename, header=None)
        csv_tmp = csv_file.copy()
        for idx, row in enumerate(csv_file.itertuples(index=True, name='Pandas')):
            lab_idx = [l for l in eval(row[3]) if l not in labels_to_delete_index]
            lab_name = [l for l in eval(row[2]) if l not in labels_to_delete_name]
            csv_tmp.at[idx, 1] = lab_name
            csv_tmp.at[idx, 2] = lab_idx
            print(idx)
        print("write csv file without the labels that are no more useful....")
        csv_tmp.to_csv(new_csv_file, header=False, index=False)
        return True

    @staticmethod
    def change_labels_in_csv(csv_filename: str, new_csv_file: str) -> True:
        """
        Function that changes the existing labels to the new ones
        :param csv_filename: intermediate csv file
        :param new_csv_file: new csv file
        :return: True
        """
        csv_file = pd.read_csv(csv_filename, header=None)
        csv_tmp = csv_file.copy()
        for idx, row in enumerate(csv_file.itertuples(index=True, name='Pandas')):
            lab_idx = [new_labels_assignments[l] for l in eval(row[3])]
            lab_name = [BigEarthNet19_labels[l] for l in lab_idx]
            csv_tmp.at[idx, 1] = lab_name
            csv_tmp.at[idx, 2] = lab_idx
            print(idx)
        print("write csv file without the labels that are no more useful....")
        csv_tmp.to_csv(new_csv_file, header=False, index=False)

    @staticmethod
    def search_empty_labels(csv_filename: str, new_csv_file: str) -> True:
        """
        Function that searches for eventual satellitary images with no labels and erases it
        :param csv_filename: intermediate csv file
        :param new_csv_file: new csv file
        :return: True
        """
        csv_file = pd.read_csv(csv_filename, header=None)
        csv_tmp = csv_file.copy()
        idx_to_delete = []
        for idx, row in enumerate(csv_file.itertuples(index=True, name='Pandas')):
            lab_idx = [new_labels_assignments[l] for l in eval(row[3])]
            if not lab_idx:
                idx_to_delete.append(idx)
            print(idx)
        print("Number of entries to delete: ", len(idx_to_delete))
        csv_tmp = csv_tmp.drop(idx_to_delete)
        print("write csv file without the empty labels entries...")
        csv_tmp.to_csv(new_csv_file, header=False, index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='BigEarthNet change labels')
    argparser.add_argument('--csv_filename', type=str, default='BigEarth.csv', help='Name of the csv dataset file')
    argparser.add_argument('--new_path_csv', type=str, default=None, help='indicate the new name of the csv')
    args = argparser.parse_args()
    Labels.delete_labels_in_csv(args.csv_filename, args.new_path_csv)
    Labels.change_labels_in_csv(args.new_path_csv, args.new_path_csv)
    Labels.search_empty_labels(args.new_path_csv, args.new_path_csv)



