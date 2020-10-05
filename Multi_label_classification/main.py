from __future__ import print_function

import argparse
import copy
import logging
import os
import warnings

import torch.nn as nn
import torch.utils.data
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from Colorization import utils
from Multi_label_classification.dataset.dataset_big_earth_torch_mlc import BigEarthDatasetTorchMLC
from Multi_label_classification.job_config import set_params
from Multi_label_classification.metrics.metric import metrics_def
from Multi_label_classification.models.ResnetMLC import ResNetMLC
from Multi_label_classification.test import test
from Multi_label_classification.train import train

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

    # define the sampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # define the loader
    train_loader = torch.utils.data.DataLoader(big_earth, batch_size=params.batch_size,
                                               sampler=train_sampler, num_workers=params.num_workers)
    val_loader = torch.utils.data.DataLoader(big_earth, batch_size=params.batch_size,
                                             sampler=val_sampler, num_workers=params.num_workers)
    test_loader = torch.utils.data.DataLoader(big_earth, batch_size=params.batch_size,
                                              sampler=test_sampler, num_workers=params.num_workers)

    # MODEL definition
    model = ResNetMLC(in_channels=params.input_channels, out_cls=params.out_cls, resnet_version=params.resnet_version,
                      pretrained=params.pretrained, colorization=params.load_checkpoint)
    # Colorization checkpoint
    if params.load_checkpoint:
        checkpoint = torch.load(params.path_model_dict)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # reset first layer when you want to apply colorization on all bands or RGB
        if params.change_first_conv:
            model.set_weights_conv1()
    # Checkpoint of the multi-label model
    if params.load_checkpoint_tr == 1:
        checkpoint = torch.load(params.path_model_dict_tr)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # CUDA
    model.to(device)

    # loss for multilabel classification
    loss_fn = nn.MultiLabelSoftMarginLoss()

    # OPTIMIZER
    if params.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9)

    # SCHEDULER
    if params.sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.sched_step, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.sched_milestones, gamma=0.1)

    if params.load_checkpoint_tr:
        optimizer.load_state_dict(checkpoint['optim_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        start_epoch = scheduler.last_epoch  # start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # METRICS
    metrics = metrics_def

    # save the best model
    best_avg_prec_micro = 0.0
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(params.epochs - start_epoch):
        # Training
        if params.load_checkpoint_tr:
            epoch += start_epoch
        logging.info("Starting training for {} epoch(s)".format(params.epochs))
        logging.info("Epoch {}/{}".format(epoch, params.epochs))
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, device=device, metrics=metrics)
        # validation
        if epoch % params.val_step == 0:
            logging.info("Starting test for {} epoch(s)".format(params.epochs))
            avg_pr_micro = test(model=model, test_loader=val_loader, loss_fn=loss_fn,
                                device=device, metrics=metrics)
            # save best model params based on avg_pr_micro score on validation set
            if avg_pr_micro > best_avg_prec_micro:
                best_avg_prec_micro = avg_pr_micro
                best_model = copy.deepcopy(model.state_dict())
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict(),
                         'scheduler_dict': scheduler.state_dict()}
                path_to_save_chk = params.path_nas + params.log_dir
                utils.save_checkpoint(state,
                                      is_best=True,  # True if this is the model with best metrics
                                      checkpoint=path_to_save_chk)  # path to folder
        # scheduler step
        if params.scheduler:
            scheduler.step()
        logging.info("lr: {}".format(scheduler.get_lr()[0]))
        # Save checkpoint
        if epoch % params.save_checkpoint == 0:
            # as I don't have a good metric to check I save the final state of the model..
            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optim_dict': optimizer.state_dict(),
                     'scheduler_dict': scheduler.state_dict()}
            path_to_save_chk = params.path_nas + params.log_dir
            utils.save_checkpoint(state,
                                  is_best=False,  # True if this is the model with best metrics
                                  checkpoint=path_to_save_chk)  # path to folder

    logging.info("Starting final test...")
    test(model=model, test_loader=test_loader, loss_fn=loss_fn, device=device, metrics=metrics)

    logging.info("Starting final test with best model...")
    model.load_state_dict(best_model)
    test(model=model, test_loader=test_loader, loss_fn=loss_fn, device=device, metrics=metrics)

    # CLOSE THE WRITER
    writer.close()


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='multi_label_classification')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--json_config_file', default='config/configuration.json', help='name of the json config file')
    parser.add_argument('--id_optim', default=1, type=int, help='id_optim parameter')
    # read the args
    args = parser.parse_args()
    main(args)
