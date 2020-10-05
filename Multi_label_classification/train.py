import logging
import time

import torch
from tqdm import tqdm

from Multi_label_classification.metrics.metric import average_precision


def train(model, train_loader, loss_fn, optimizer, device, metrics):
    start_time = time.time()
    # SET THE MODEL TO TRAIN MODE
    model.train()

    train_loss = 0

    # EPOCH METRICS
    out_sigmoid_epoch = torch.Tensor()
    labels_class_epoch = torch.Tensor()

    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (in_bands, labels_class) in enumerate(train_loader):
            # move input data to GPU
            in_bands = in_bands.to(device)
            labels_class = labels_class.to(device)

            # set the gradient to zero
            optimizer.zero_grad()

            # FORWARD PASS
            out, out_sigmoid = model(in_bands)
            # multi-label classification loss
            loss = loss_fn(out, labels_class)

            # BACKWARD PASS
            loss.backward()

            train_loss += loss.item()

            # write loss
            t.set_postfix(loss='{:05.3f}'.format(loss.item()))
            t.update()

            # update the params of the model
            optimizer.step()

            # concat tensor in order to calculate the overall metrics for the entire epoch
            labels_class_epoch = torch.cat((labels_class_epoch, labels_class.type(torch.FloatTensor).cpu()), 0)
            out_sigmoid_epoch = torch.cat((out_sigmoid_epoch, out_sigmoid.cpu()), 0)

        # Threshold of 0.5 on the sigmoid output for the entire epoch
        out_thresholded_epoch = (out_sigmoid_epoch > 0.5).float()

        # overall metrics for the current epoch, log on file
        metrics_calc = {metric: metrics[metric](labels_class_epoch.cpu(), out_thresholded_epoch.cpu(), 'micro') for metric in metrics}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
        logging.info("- Train metrics : " + metrics_string)

        # MAP
        avg_pr_micro = average_precision(labels_class_epoch.cpu(), out_sigmoid_epoch.cpu(), average='micro')
        avg_pr_macro = average_precision(labels_class_epoch.cpu(), out_sigmoid_epoch.cpu(), average='macro')
        avg_pr_weighted = average_precision(labels_class_epoch.cpu(), out_sigmoid_epoch.cpu(), average='weighted')
        avg_pr_none = average_precision(labels_class_epoch.cpu(), out_sigmoid_epoch.cpu(), average=None)
        dict_avg_pr_none = dict(zip(list(range(19)), avg_pr_none))

        logging.info("\n- Train MAP : micro: {:05.3f} ; macro: {:05.3f}; weighted: {:05.3f}".format(avg_pr_micro, avg_pr_macro, avg_pr_weighted))
        avg_pr_none_str = "\n".join("cls: {} --> val: {:05.3f} ".format(cls, val) for cls, val in dict_avg_pr_none.items())
        logging.info("Train MAP: None\n" + avg_pr_none_str)

        time_elapsed = time.time() - start_time
        logging.info('Epoch complete in {:.0f}m {:.0f}s. Avg training loss: {:05.3f}'.format(
            time_elapsed // 60, time_elapsed % 60, train_loss / len(train_loader)))
