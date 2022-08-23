import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import sys
from collections import OrderedDict

from core.torchplus import Sequential, Empty, change_default_args
from core.torchplus import kaiming_init
from core.models.biattention_conv2d_concat_initW import AttentionConv2D

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, padding = 0, dilation = 1, groups = 1, pooling_r = 4, norm_layer = Empty):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 layer_nums=(8, 8),
                 layer_strides=(1, 2),
                 num_filters=(32, 64),
                 upsample_strides=(1, 2),
                 num_upsample_filters=(64, 64),
                 num_input_features=35):

        super(RPN, self).__init__()

        assert len(layer_nums) == 2
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        # upsample_strides = [
        #     np.round(u).astype(np.int64) for u in upsample_strides
        # ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)


        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        block1 = [
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        ]
        for i in range(layer_nums[0]):
            block1.append(
                SCConv(num_filters[0], num_filters[0], padding=1, norm_layer=BatchNorm2d))
            #self.block1.add(BatchNorm2d(num_filters[0]))
            block1.append(nn.ReLU())
        deconv1 = [
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        ]
        block2 = [
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        ]
        for i in range(layer_nums[1]):
            block2.append(
                SCConv(num_filters[1], num_filters[1], padding=1, norm_layer=BatchNorm2d))
            block2.append(nn.ReLU())
        deconv2 = [
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        ]

        self.block1 = nn.Sequential(*block1)
        self.deconv1 = nn.Sequential(*deconv1)
        self.block2 = nn.Sequential(*block2)
        self.deconv2 = nn.Sequential(*deconv2)
        

    def forward(self, x):
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        
        x = torch.cat([up1, up2], dim=1)
        return x

class Head(nn.Module):
    def __init__(self, out):
        super(Head, self).__init__()
        self.conv = conv3x3(128, 64)
        self.head = nn.Conv2d(64, out, kernel_size=1)


    def forward(self, x):
        x = self.conv(x)
        head = self.head(x)
         
        return head

class Header(nn.Module):
    def __init__(self, num_classes):
        super(Header, self).__init__()

        self.cls = Head(num_classes)
        self.offset = Head(2)
        self.size = Head(2)
        self.yaw = Head(2)


    def forward(self, x):
        cls = self.cls(x)
        cls = torch.sigmoid(cls)
        offset = self.offset(x)
        size = self.size(x)
        yaw = self.yaw(x) 

        return cls, offset, size, yaw

class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        attention_flag=True,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)
        self.heads = heads
        self._heatmap = None
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            if attention_flag:
                for i in range(num_conv-1):
                    fc.add(AttentionConv2D(in_channels, head_conv,
                                     kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True,
                                     is_header=True, last_attention=True))
            else:
                for i in range(num_conv-1):
                    fc.add(nn.Conv2d(in_channels, head_conv,
                                     kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True))
            if bn and (not attention_flag):
                fc.add(nn.BatchNorm2d(head_conv))
            fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1,
                    padding=final_kernel // 2, bias=True))

            # init weight.
            '''
            elif 'occlusion' in head:
                normal_init(fc[-1], std=0.01)
            '''

            if 'cls' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                # init attention weight.
                for m in fc.modules():
                    if isinstance(m, AttentionConv2D):
                        m.init_attention_weight()

            # setattr() is used to assign the object attribute its value
            # object (self), name (head / 'reg' like key), value (fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        # self._heatmap = ret_dict['hm']
        return ret_dict
 
    @property
    def hm_attention(self):
        return self.__getattr__('cls')[0].att_map

    @property
    def heatmap(self):
        return torch.clamp(
            self._heatmap.sigmoid_(), min=1e-4, max=1-1e-4)


class AFDet(nn.Module):
    def __init__(self, num_classes = 4):
        super(AFDet, self).__init__()
        self.backbone = RPN()
        self.header = Header(num_classes)

    def forward(self, x, x_min: float, y_min: float, x_res: float, y_res: float, score_threshold: float):
        features = self.backbone(x)
        cls, offset, size, yaw = self.header(features)

        offset_pred = offset[0].detach()
        size_pred = size[0].detach()
        yaw_pred = yaw[0].detach()
        cls_pred = cls[0].detach()

        cos_t, sin_t = torch.chunk(yaw_pred, 2, dim = 0)
        dx, dy = torch.chunk(offset_pred, 2, dim = 0)
        log_w, log_l = torch.chunk(size_pred, 2, dim = 0)

        cls_probs, cls_ids = torch.max(cls_pred, dim = 0)

    
        pooled = F.max_pool2d(cls_probs.unsqueeze(0), 3, 1, 1).squeeze()
        selected_idxs = torch.logical_and(cls_probs == pooled, cls_probs > score_threshold)

        y = torch.arange(cls.shape[2])
        x = torch.arange(cls.shape[3])

        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xx = xx.to(offset_pred.device)
        yy = yy.to(offset_pred.device)

        center_y = dy + yy *  y_res + y_min
        center_x = dx + xx *  x_res + x_min
        center_x = center_x.squeeze()
        center_y = center_y.squeeze()
        l = torch.exp(log_l).squeeze()
        w = torch.exp(log_w).squeeze()
        yaw2 = torch.atan2(sin_t, cos_t).squeeze()
        yaw = yaw2 / 2

        boxes = torch.cat([cls_ids[selected_idxs].reshape(-1, 1), 
                        cls_probs[selected_idxs].reshape(-1, 1), 
                        center_x[selected_idxs].reshape(-1, 1), 
                        center_y[selected_idxs].reshape(-1, 1), 
                        l[selected_idxs].reshape(-1, 1), 
                        w[selected_idxs].reshape(-1, 1), 
                        yaw[selected_idxs].reshape(-1, 1)], dim = 1)

        return boxes


        


class RAAFDet(nn.Module):
    def __init__(self):
        super(RAAFDet, self).__init__()
        heads = {"cls": [4, 2],
                 "offset": [2, 2],
                 "size": [2, 2],
                 "yaw": [2, 2]}

        self.tasks = SepHead(in_channels = 128, heads = heads)
        self.backbone = RPN()

    def forward(self, x):
        features = self.backbone(x)
        pred = self.tasks(features)
        pred["cls"] = self._sigmoid(pred["cls"])
        return pred    

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y


if __name__ == "__main__":
    model = RAAFDet()
    x = torch.rand([1, 35, 400, 400])
    #model.to("cuda:0")
    pred = model(x)
    print(pred["cls"].shape)
    #print("number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))