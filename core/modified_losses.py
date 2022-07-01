import torch
import torch.nn as nn
import torch.nn.functional as F

def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds] + 1e-7
    neg_pred = pred[neg_inds] - 1e-7
    
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss




def _l1_loss(pred, target, mask):
    mask = mask.unsqueeze(1).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss



class AFDetLoss(nn.Module):
    def __init__(self):
        super(AFDetLoss, self).__init__()

    def forward(self, pred, target):
        cls_loss = _slow_neg_loss(pred["cls"], target["cls"])
        offset_loss = _l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = _l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = _l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])

        loss = cls_loss + offset_loss + size_loss + yaw_loss

        return loss, cls_loss.item(), offset_loss.item(), size_loss.item(), yaw_loss.item() 