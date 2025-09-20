import torch
import torch.nn as nn


"""
class-conditional feature moment matching (CFMM) loss function
"""

class MCFMLoss(nn.Module):
    def __init__(self, num_classes, wt=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.wt = wt

    def CoralLoss(self, Xs, Xt):
        d = Xs.data.shape[1]
        ns, nt = Xs.data.shape[0], Xt.data.shape[0]

        xm = torch.mean(Xs, 0, keepdim=True) - Xs
        xc = xm.t() @ xm / (ns - 1)

        xmt = torch.mean(Xt, 0, keepdim=True) - Xt
        xct = xmt.t() @ xmt / (nt - 1)

        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss

    def FMLoss(self, Xs, Xt):
        diff = torch.mean(Xs, 0) - torch.mean(Xt, 0)
        loss = torch.mean(torch.abs(diff))
        return loss

    def forward(self, source_fea, source_label, tar_fea, tar_label):
        device = source_fea.device
        valid_classes = 0
        total_loss = torch.tensor(0.0, device=device)

        for c in range(self.num_classes):
            src_mask = (source_label == c)
            src_features_c = source_fea[src_mask]

            tgt_mask  = (tar_label == c)
            tgt_features_c  = tar_fea[tgt_mask]

            if src_features_c.shape[0] < 2 or tgt_features_c.shape[0] < 2:
                continue

            coral_loss = self.CoralLoss(src_features_c, tgt_features_c)
            fm_loss = self.FMLoss(src_features_c, tgt_features_c)
            total_loss += self.wt * coral_loss + fm_loss
            valid_classes += 1

        if valid_classes == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_classes
