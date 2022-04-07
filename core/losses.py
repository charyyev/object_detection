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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, cls_target: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((cls_target.shape))
        mask = mask.to("cuda:0")
        #print(input.get_device())
        mask[cls_target > 0] = 1
        mask = torch.unsqueeze(mask, 1).repeat(1, input.shape[1], 1, 1)
        # print(mask.shape)
        # print(input.shape)
        # print(target.shape)
        #mask.to(input.get_device())
        loc_loss = F.smooth_l1_loss(input * mask, target * mask, reduction='mean') 

        return loc_loss




if __name__ == "__main__":
    loss = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")
    x = torch.zeros((1, 3, 200, 175))
    y = torch.zeros((1, 200, 175), dtype=torch.int64)

    x[0][0][:, :] = 20
    print(x)

    print(loss(x, y))

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