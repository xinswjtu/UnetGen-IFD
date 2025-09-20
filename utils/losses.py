import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:  # for D
            loss_real = F.relu(1. - pred_real).mean()
            loss_fake = F.relu(1. + pred_fake).mean()
            loss = loss_real + loss_fake
            return loss
        else:  # for G
            loss = -pred_real.mean()
            return loss


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, cls_output, label, **_):
        return self.ce_loss(cls_output, label)



