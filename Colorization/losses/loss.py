import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, mode="L1"):
        super().__init__()
        self.mode = mode
        if self.mode == 'L1':
            self.loss = nn.L1Loss(reduction='mean')  
        else:
            self.loss = nn.MSELoss(reduction='mean')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        return self.loss(pred, gt)

    @staticmethod
    def grad_loss_fn(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = gt.shape
        # Horinzontal Sobel filter
        Sx = torch.Tensor(([-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1])).cuda()
        # reshape the filter and compute the conv
        Sx = Sx.expand(1, C, 3, 3)
        Gt_x = F.conv2d(gt, Sx, padding=1)
        pred_x = F.conv2d(pred, Sx, padding=1)
        # Vertical Sobel filter
        Sy = torch.Tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]]).cuda()
        # reshape the filter and compute the conv
        Sy = Sy.expand(1, C, 3, 3)
        Gt_y = F.conv2d(gt, Sy, padding=1)
        pred_y = F.conv2d(pred, Sy, padding=1)

        loss_grad_x = torch.pow((Gt_x - pred_x), 2).mean()
        loss_grad_y = torch.pow((Gt_y - pred_y), 2).mean()
        loss_grad = loss_grad_x + loss_grad_y
        return loss_grad



