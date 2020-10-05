import logging
import time

import torch
from Colorization.dataset.dataset_big_earth import Color
from tqdm import tqdm


def test(model, test_loader, loss_fn, device, params, epoch, writer):
    start_time = time.time()
    # SET THE MODEL TO EVALUATION MODE
    model.eval()

    test_loss = 0.

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, (spectral, L, ab) in enumerate(test_loader):
                # move input data to GPU
                spectral = spectral.to(device)
                ab = ab.to(device)

                # FORWARD PASS
                out = model(spectral=spectral)
                loss = loss_fn(out, ab) * params.weight_rec_loss

                # GRAD LOSS
                if params.grad_loss:
                    grad_loss = loss_fn.grad_loss_fn(out, ab)*params.weight_grad_loss
                    loss += grad_loss

                test_loss += loss.item()

                # write loss
                t.set_postfix(loss='{:05.3f}'.format(loss.item()))
                t.update()

                # log on tensorboard
                if batch_idx % params.log_interval == 0:
                    # LOSS LOG
                    writer.add_scalar('Tot_Loss/test', loss.item(), epoch * len(test_loader) + batch_idx)
                    if params.grad_loss:
                        writer.add_scalar('Grad_Loss/test', grad_loss.item(), epoch * len(test_loader) + batch_idx)
                    # IMAGE LOG
                    n = min(out.size(0), 8)
                    recon_rgb_n = Color.lab2rgb(L[:n], out[:n])
                    rgb_n = Color.lab2rgb(L[:n], ab[:n])
                    comparison = torch.cat([rgb_n[:n], recon_rgb_n[:n]])
                    writer.add_images('comparison_original_recon/test', comparison,
                                      epoch * len(test_loader) + batch_idx)

            time_elapsed = time.time() - start_time
            logging.info('Test complete in {:.0f}m {:.0f}s. Avg test loss: {:05.3f}'.format(
                time_elapsed // 60, time_elapsed % 60, test_loss / len(test_loader)))

