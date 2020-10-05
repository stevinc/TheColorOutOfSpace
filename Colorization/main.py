from __future__ import print_function

import argparse
import logging
import os
import warnings

import numpy as np
import torch.utils.data
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from Colorization import utils
from Colorization.dataset.dataset_big_earth_torch import BigEarthDatasetTorch
from Colorization.job_config import set_params
from Colorization.losses.loss import Loss
from Colorization.models.Resnet18 import ResNet18
from Colorization.models.Resnet50 import ResNet50
from Colorization.test import test
from Colorization.train import train

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def init_worker(id_worker):
    np.random.seed(42 + id_worker)


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
    writer = SummaryWriter(params.path_nas + params.log_dir + '/runs/' + params.log_dir)

    # create dir for log file
    if not os.path.exists(params.path_nas + params.log_dir):
        os.makedirs(params.path_nas + params.log_dir)
    # save the json config file of the model
    params.save(os.path.join(params.path_nas + params.log_dir, "params.json"))

    # Set the logger
    utils.set_logger(os.path.join(params.path_nas + params.log_dir, "log"))

    # DATASET
    # Torch version
    big_earth = BigEarthDatasetTorch(csv_path=params.dataset, random_seed=params.seed, bands_indices=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                     img_size=params.img_size, augmentation=params.augmentation, n_samples=params.dataset_nsamples)

    train_idx, val_idx, test_idx = big_earth.split_dataset(params.test_split, params.val_split)
    # define the sampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # define the loader
    train_loader = torch.utils.data.DataLoader(big_earth, batch_size=params.batch_size,
                                               sampler=train_sampler, num_workers=params.num_workers, worker_init_fn=init_worker)
    val_loader = torch.utils.data.DataLoader(big_earth, batch_size=params.batch_size,
                                             sampler=val_sampler, num_workers=params.num_workers, worker_init_fn=init_worker)
    test_loader = torch.utils.data.DataLoader(big_earth, batch_size=params.batch_size,
                                              sampler=test_sampler, num_workers=params.num_workers, worker_init_fn=init_worker)
    # MODEL definition
    if params.backbone == 50:
        model = ResNet50(in_channels=params.input_channels, out_channels=params.out_channels, pretrained=params.pretrained,
                         dropout=params.dropout, decoder_version=params.decoder_version)
    else:
        model = ResNet18(in_channels=params.input_channels, out_channels=params.out_channels, pretrained=params.pretrained,
                         dropout=params.dropout)

    # eventually load checkpoint
    if params.load_checkpoint == 1:
        checkpoint = torch.load(params.path_model_dict)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # CUDA
    model.to(device)

    # LOSS ON RECONSTRUCTION
    loss_fn = Loss(mode=params.loss)

    # OPTIMIZER
    if params.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)

    # SCHEDULER
    if params.sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.sched_step, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.sched_milestones, gamma=0.1)

    if params.load_checkpoint:
        optimizer.load_state_dict(checkpoint['optim_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])

    for epoch in range(params.epochs-start_epoch):
        # Training
        if params.load_checkpoint:
            epoch += start_epoch

        logging.info("Epoch {}/{}".format(epoch, params.epochs))

        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer,
              device=device, params=params, epoch=epoch, writer=writer)
        # validation
        if epoch % params.val_step == 0:
            logging.info("Starting test for {} epoch(s)".format(params.epochs))
            test(model=model, test_loader=val_loader, loss_fn=loss_fn,
                 device=device, params=params, epoch=epoch, writer=writer)
        # scheduler step
        if params.scheduler:
            scheduler.step()
        # Save checkpoint
        if epoch % params.save_checkpoint == 0:
            if params.scheduler:
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict(),
                         'scheduler_dict': scheduler.state_dict()}
            else:
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()}
            path_to_save_chk = params.path_nas + params.log_dir
            utils.save_checkpoint(state,
                                  is_best=False,
                                  checkpoint=path_to_save_chk)

    logging.info("Starting final test...")
    test(model=model, test_loader=test_loader, loss_fn=loss_fn,
         device=device, params=params, epoch=1, writer=writer)

    # CLOSE THE WRITER
    writer.close()


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='Colorization')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--json_config_file', default='../Colorization/config/configuration.json', help='name of the json config file')
    parser.add_argument('--id_optim', default=0, type=int, help='id_optim parameter')
    # read the args
    args = parser.parse_args()
    main(args)




