from __future__ import print_function

import argparse
import logging
import os
import warnings

import torch.nn as nn
import torch.utils.data
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from Colorization import utils
from Multi_label_classification.dataset.dataset_big_earth_torch_mlc import BigEarthDatasetTorchMLC
from Multi_label_classification.job_config import set_params
from Multi_label_classification.metrics.metric import metrics_def
from Multi_label_classification.models.Ensemble import EnsembleModel
from Multi_label_classification.models.ResnetMLC import ResNetMLC
from Multi_label_classification.test import test

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def main(args):
    # enable cuda if available
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # READ JSON CONFIG FILE
    assert os.path.isfile(args.json_config_file), "No json configuration file found at {}".format(args.json_config_file)
    params = utils.Params(args.json_config_file)

    # for change params related to job-id
    params = set_params(params, args.id_optim)

    # set the torch seed
    torch.manual_seed(params.seed)

    # initialize summary writer; every folder is saved inside runs
    writer = SummaryWriter(params.path_nas + params.log_dir + '/runs/')

    # create dir for log file
    if not os.path.exists(params.path_nas + params.log_dir):
        os.makedirs(params.path_nas + params.log_dir)

    # save the json config file of the model
    params.save(os.path.join(params.path_nas + params.log_dir, "params.json"))

    # Set the logger
    utils.set_logger(os.path.join(params.path_nas + params.log_dir, "log"))

    # DATASET
    # Torch version
    big_earth = BigEarthDatasetTorchMLC(csv_path=params.dataset, random_seed=params.seed, bands_indices=params.bands,
                                        img_size=params.img_size, n_samples=params.dataset_nsamples)
    # Split
    train_idx, val_idx, test_idx = big_earth.split_dataset(params.test_split, params.val_split)

    test_sampler = SubsetRandomSampler(test_idx)
    # define the loader
    test_loader = torch.utils.data.DataLoader(big_earth, batch_size=params.batch_size,
                                              sampler=test_sampler, num_workers=params.num_workers)
    # MODELS definition for Ensemble
    model_rgb = ResNetMLC(in_channels=3, out_cls=params.out_cls, resnet_version=params.resnet_version,
                          pretrained=0, colorization=0)
    model_colorization = ResNetMLC(in_channels=9, out_cls=params.out_cls, resnet_version=params.resnet_version,
                                   pretrained=0, colorization=1)

    checkpoint = torch.load(args.rgb_checkpoint)
    model_rgb.load_state_dict(checkpoint['state_dict'], strict=False)

    checkpoint = torch.load(args.spectral_checkpoint)
    model_colorization.load_state_dict(checkpoint['state_dict'], strict=False)

    model = EnsembleModel(model_rgb=model_rgb, model_colorization=model_colorization, device=device)

    # CUDA
    model.to(device)

    # loss for multilabel classification
    loss_fn = nn.MultiLabelSoftMarginLoss()

    # METRICS
    metrics = metrics_def

    logging.info("Starting final test with ensemble model...")
    test(model=model, test_loader=test_loader, loss_fn=loss_fn,
         device=device, metrics=metrics)

    # CLOSE THE WRITER
    writer.close()


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='multi_label_classification')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--json_config_file', default='Multi_label_classification/config/configuration.json', help='name of the json config file')
    parser.add_argument('--id_optim', default=0, type=int, help='id_optim parameter')
    parser.add_argument('--rgb_checkpoint', type=str, default=None, help='specify the rgb checkpoint path', required=True)
    parser.add_argument('--spectral_checkpoint', type=str, default=None, help='specify the spectral checkpoint path', required=True)
    # read the args
    args = parser.parse_args()
    main(args)
