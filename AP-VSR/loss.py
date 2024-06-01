"""
This code is based on Open-MMLab's one.
https://github.com/open-mmlab/mmediting
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

def charbonnier_loss(pred, target,weight=None,reduction='mean',sample_wise=False, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    loss = torch.sqrt((pred - target)**2 + eps)

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss

    # return torch.sqrt((pred - target)**2 + eps).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f"Supported ones are: ['none', 'mean', 'sum']")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


class FFTLoss(nn.Module):
    def __init__(self, lambda_param=1.0):
        super(FFTLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, predicted_frame, ground_truth_frame):
        predicted_fft = torch.fft.fft2(predicted_frame)
        gt_fft = torch.fft.fft2(ground_truth_frame)

        amplitude_loss = torch.sqrt((torch.abs(predicted_fft) - torch.abs(gt_fft)) ** 2 + 1e-12).mean()
        phase_loss = torch.sqrt((torch.angle(predicted_fft) - torch.angle(gt_fft)) ** 2 + 1e-12).mean()
        # amplitude_loss = F.l1_loss(torch.abs(predicted_fft), torch.abs(gt_fft),reduction='mean')
        # phase_loss = F.l1_loss(torch.angle(predicted_fft), torch.angle(gt_fft),reduction='mean')
        # amplitude_loss = F.mse_loss(torch.abs(predicted_fft), torch.abs(gt_fft),reduction='mean')
        # phase_loss = F.mse_loss(torch.angle(predicted_fft), torch.angle(gt_fft),reduction='mean')

        fft_loss = amplitude_loss + self.lambda_param * phase_loss
        return fft_loss

from pytorch_msssim import ssim
class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_true, y_pred):
        ssim_val = ssim(y_true, y_pred,data_range=1.0,size_average=True)
        return ssim_val


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 1e-3

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        # assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1).cpu()
        label_onehot = torch.zeros([batch_size, labels_length]).scatter_(1, new_label, 1).cuda()

        # calculate log
        log_p = F.log_softmax(logits+self.elipson)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]


        # print(logits)
        logits = torch.clamp(logits, min=1e-7, max=1 - 1e-7)
        logits = F.log_softmax(logits, 1)
        # print(logits)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



