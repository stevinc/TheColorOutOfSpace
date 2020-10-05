import logging
import time

import torch
from Colorization.dataset.dataset_big_earth import Color
from tqdm import tqdm


def train(model, train_loader, loss_fn, optimizer, device, params, epoch, writer):
    start_time = time.time()
    # SET THE MODEL TO TRAIN MODE
    model.train()

    train_loss = 0.
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (spectral, L, ab) in enumerate(train_loader):
            # move input data to GPU
            spectral = spectral.to(device)
            ab = ab.to(device)
            # set the gradient to zero
            optimizer.zero_grad()
            # FORWARD PASS
            out = model(spectral=spectral)
            # L2 | L1 LOSS ON RECONSTRUCTION
            loss = loss_fn(out, ab) * params.weight_rec_loss
            # GRAD LOSS
            if params.grad_loss:
                grad_loss = loss_fn.grad_loss_fn(out, ab) * params.weight_grad_loss
                loss += grad_loss
            # BACKWARD PASS
            loss.backward()
            train_loss += loss.item()
            # write loss
            t.set_postfix(loss='{:05.3f}'.format(loss.item()))
            t.update()
            # update the params of the model
            optimizer.step()
            # log on tensorboard
            if batch_idx % params.log_interval == 0:
                # LOSS LOG
                writer.add_scalar('Total_Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
                if params.grad_loss:
                    writer.add_scalar('Grad_Loss/train', grad_loss.item(), epoch * len(train_loader) + batch_idx)
                # IMAGE LOG
                n = min(out.size(0), 8)
                recon_rgb_n = Color.lab2rgb(L[:n], out[:n])
                rgb_n = Color.lab2rgb(L[:n], ab[:n])
                comparison = torch.cat([rgb_n[:n], recon_rgb_n[:n]])
                writer.add_images('comparison_original_recon/train', comparison, epoch * len(train_loader) + batch_idx)

        time_elapsed = time.time() - start_time
        logging.info('Epoch complete in {:.0f}m {:.0f}s. Avg training loss: {:05.3f}'.format(
            time_elapsed // 60, time_elapsed % 60, train_loss / len(train_loader)))
