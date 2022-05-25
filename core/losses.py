import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.one_hot import one_hot

# taken from https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
# https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
# https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    alphas = [0.05, 0.15, 0.8, 0.8]
) -> torch.Tensor:

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)
    focal = -alpha * weight * log_input_soft
    #focal = -log_input_soft
    if alphas is not None:
        for i in range(len(alphas)):
            focal[:, i, ...] *= alphas[i] 
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.alpha: float = config["alpha"]
        self.gamma: float = config["gamma"]
        self.reduction: str = config["reduction"]


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction)


class SmoothL1Loss(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor, target: torch.Tensor, cls_target: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((cls_target.shape), dtype = torch.int16)
        mask = mask.to(self.device)
        
        mask[cls_target > 0] = 1
        num_pixels = torch.sum(mask)
        mask = torch.unsqueeze(mask, 1).repeat(1, input.shape[1], 1, 1)
        #loc_loss = F.smooth_l1_loss(input * mask, target * mask, reduction='sum') / num_pixels
        if num_pixels <= 0:
            return torch.tensor([0]).to(self.device)
        loc_loss = F.smooth_l1_loss(input[mask == 1], target[mask == 1], reduction='mean')
        return loc_loss


class CustomLoss(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.alpha: float = config["alpha"]
        self.gamma: float = config["gamma"]
        self.reduction: str = config["reduction"]
        self.device = device

    def forward(self, pred, target):
        cls_pred = pred["cls_map"]
        reg_pred = pred["reg_map"]
        cls_target = target["cls_map"]
        reg_target = target["reg_map"]
        submap_target = target["sub_map"]

        #submap_pred = torch.unsqueeze(submap_target, 1).repeat(1, cls_pred.shape[1], 1, 1)
        #cls = cls_pred[submap_pred == 1].view((1, cls_pred.shape[1], -1))
        idxs = submap_target == 1
        num = idxs.nonzero().shape[0]
        cls = torch.zeros(1, cls_pred.shape[1], num)
        cls = cls.to(self.device)
        for i in range(cls_pred.shape[1]):
            cls[0, i, :] = cls_pred[:, i, :, :][idxs]
       
        fc_loss = focal_loss(cls, cls_target[submap_target == 1].view(1, -1), self.alpha, self.gamma, self.reduction)
      
        mask = torch.zeros(cls_target.shape, dtype = torch.int16)
        mask = mask.to(self.device)
        
        #mask[cls_target > 0] = 1
        #mask[submap_target == 1] = 1
        mask[torch.logical_and(cls_target > 0, submap_target == 1)] = 1
        mask = torch.unsqueeze(mask, 1).repeat(1, reg_pred.shape[1], 1, 1)
        num_pixels = torch.sum(mask)

        #loc_loss = F.smooth_l1_loss(reg_pred * mask, reg_target * mask, reduction='mean')
        #loss = loc_loss + fc_loss
        if num_pixels <= 0:
            loss = fc_loss
            loc_loss = torch.tensor([0])
        else:
            loc_loss = F.smooth_l1_loss(reg_pred[mask == 1], reg_target[mask == 1], reduction='mean')
            loss = 2 * fc_loss + loc_loss

        return loss, fc_loss.item(), loc_loss.item()

class HotSpotLoss(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.alpha: float = config["alpha"]
        self.gamma: float = config["gamma"]
        self.reduction: str = config["reduction"]
        self.device = device

    def forward(self, pred, target):
        cls_pred = pred["cls_map"]
        reg_pred = pred["reg_map"]
        quad_pred = pred["quad_map"]
        xargmin_pred = pred["x"]
        yargmin_pred= pred["y"]
        cls_target = target["cls_map"]
        reg_target = target["reg_map"]
        hotspot_mask = target["hotspot_mask"]
        xargmin_target = target["xargmin"]
        yargmin_target = target["yargmin"]
        quad_target = target["quad_map"]

        idxs = hotspot_mask == 1
        num = idxs.nonzero().shape[0]
        cls = torch.zeros(1, cls_pred.shape[1], num)
        cls = cls.to(self.device)
        for i in range(cls_pred.shape[1]):
            cls[0, i, :] = cls_pred[:, i, :, :][idxs]
       
        cls_loss = focal_loss(cls, cls_target[hotspot_mask == 1].view(1, -1), self.alpha, self.gamma, self.reduction)

        hotspot_idxs = (cls_target * hotspot_mask) > 0
        hotspot_num = hotspot_idxs.nonzero().shape[0]
        quad = torch.zeros(1, quad_pred.shape[1], hotspot_num)
        xargmin = torch.zeros(1, xargmin_pred.shape[1], hotspot_num)
        yargmin = torch.zeros(1, yargmin_pred.shape[1], hotspot_num)

        quad = quad.to(self.device)
        xargmin = xargmin.to(self.device)
        yargmin = yargmin.to(self.device)


        for i in range(quad.shape[1]):
            quad[0, i, :] = quad_pred[:, i, :, :][hotspot_idxs]
        for i in range(xargmin.shape[1]):
            xargmin[0, i, :] = xargmin_pred[:, i, :, :][hotspot_idxs]
        for i in range(yargmin.shape[1]):
            yargmin[0, i, :] = yargmin_pred[:, i, :, :][hotspot_idxs]

        quad_loss = focal_loss(quad, quad_target[hotspot_idxs].view(1, -1), self.alpha, self.gamma, self.reduction, None)
        xargmin_loss = focal_loss(xargmin, xargmin_target[hotspot_idxs].view(1, -1), self.alpha, self.gamma, self.reduction, None)
        yargmin_loss = focal_loss(yargmin, yargmin_target[hotspot_idxs].view(1, -1), self.alpha, self.gamma, self.reduction, None)
      
        mask = torch.zeros(cls_target.shape, dtype = torch.int16)
        mask = mask.to(self.device)
        
        mask[torch.logical_and(cls_target > 0, hotspot_mask == 1)] = 1
        mask = torch.unsqueeze(mask, 1).repeat(1, reg_pred.shape[1], 1, 1)
        num_pixels = torch.sum(mask)


        if num_pixels <= 0:
            loss = cls_loss
            loc_loss = torch.tensor([0])
        else:
            loc_loss = F.smooth_l1_loss(reg_pred[mask == 1], reg_target[mask == 1], reduction='mean') + xargmin_loss + yargmin_loss
            loss = cls_loss + loc_loss + quad_loss

        return loss, cls_loss.item(), loc_loss.item(), quad_loss.item()

if __name__ == "__main__":
    input = torch.ones((2, 4, 5))
    alpha = torch.tensor([0.05, 0.15, 0.4, 0.4])
    for i in range(alpha.shape[0]):
        input[:, i, ...] *= alpha[i]
    print(input)
    # mask = torch.unsqueeze(input, 1).repeat(1, 4, 1, 1)
    # print(mask[0])
    # #loss = F.smooth_l1_loss(input, target, reduction = "none")
    #print(loss)


    # loss = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")
    # x = torch.zeros((1, 3, 200, 175))
    # y = torch.zeros((1, 200, 175), dtype=torch.int64)

    # x[0][0][:, :] = 20
    # print(x)

    # print(loss(x, y))

    # for i in range(200):
    #     for j in range(175):
    #         rand = torch.randint(high = 3, size =(1, 1))[0][0]
    #         x[0][i][j][rand] = 1
   
    # x = torch.tensor([[4, 2, 1.9], [0.2, 0.8, 3]])
    # t = torch.tensor([[1, 0, 0], [0, 1, 0]])
    # m = nn.Sigmoid()
    # p = x.sigmoid()
    # pt = p*t + (1-p)*(1-t)
    # print(pt)