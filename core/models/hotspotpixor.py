import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import time
from typing import Callable, Any, Optional, List


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv3x3_dw(in_planes, out_planes, stride = 1, bias = False):
    return nn.Sequential(
        # dw
        nn.Conv2d(in_planes, out_planes, 3, stride, 1, groups=in_planes, bias=bias),
        nn.BatchNorm2d(in_planes),
        nn.ReLU(inplace=True),

        # pw
        nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        )



class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BackBone(nn.Module):

    def __init__(self, block, geom, use_bn=True):
        super(BackBone, self).__init__()

        self.use_bn = use_bn

        # Block 1
        self.conv1 = conv3x3(35, 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)


        # Block 2-5
        self.input_channel = 32
        self.block2 = self._make_layer(block, 1, 24, 1, 2)
        self.block3 = self._make_layer(block, 6, 32, 3, 2)
        self.block4 = self._make_layer(block, 6, 64, 4, 2)
        self.block5 = self._make_layer(block, 6, 96, 3, 2)

        
        # self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        # self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        # self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        p = 0 if geom['label_shape'][1] == 175 else 1
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, p))

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p5 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p4 = l3 + self.deconv2(p5)

        return p4


    def _make_layer(self, block, t, c, n, s):
        #output_channel = _make_divisible(c * width_mult, round_nearest)
        output_channel = c
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(self.input_channel, output_channel, stride, expand_ratio=t))
            self.input_channel = output_channel

        return nn.Sequential(*features)




class Header(nn.Module):
    def __init__(self, use_bn=True):
        super(Header, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn
        self.conv1 = conv3x3_dw(16, 16, bias=bias)
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = conv3x3_dw(16, 16, bias=bias)
        #self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = conv3x3_dw(16, 16, bias=bias)
        #self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = conv3x3(16, 16, bias=bias)
        self.bn4 = nn.BatchNorm2d(16)

        self.clshead = conv3x3(16, 4, bias=True)
        self.reghead = conv3x3(16, 4, bias=True)
        self.quadhead = conv3x3(16, 4, bias=True)
        self.xargminhead = conv3x3(16, 16, bias=True)
        self.yargminhead = conv3x3(16, 16, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = self.clshead(x)
        reg = self.reghead(x)
        quad = self.quadhead(x)
        dx = self.xargminhead(x)
        dy = self.yargminhead(x)

        return cls, reg, quad, dx, dy


class HotSpotPIXOR(nn.Module):
    '''
    The input of PIXOR nn module is a tensor of [batch_size, height, weight, channel]
    The output of PIXOR nn module is also a tensor of [batch_size, height/4, weight/4, channel]
    Note that we convert the dimensions to [C, H, W] for PyTorch's nn.Conv2d functions
    '''

    def __init__(self, geom, use_bn=True, decode=False):
        super(HotSpotPIXOR, self).__init__()
        self.backbone = BackBone(InvertedResidual,  geom, use_bn)
        self.header = Header(use_bn)
        self.use_decode = decode
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        prior = 0.01
        self.header.clshead.weight.data.fill_(-math.log((1.0-prior)/prior))
        self.header.clshead.bias.data.fill_(0)
        self.header.reghead.weight.data.fill_(0)
        self.header.reghead.bias.data.fill_(0)
        self.header.quadhead.weight.data.fill_(0)
        self.header.quadhead.bias.data.fill_(0)
        self.header.xargminhead.weight.data.fill_(-math.log((1.0-prior)/prior))
        self.header.xargminhead.bias.data.fill_(0)
        self.header.yargminhead.weight.data.fill_(-math.log((1.0-prior)/prior))
        self.header.yargminhead.bias.data.fill_(0)


    def forward(self, x):        
        # Torch Takes Tensor of shape (Batch_size, channels, height, width)

        features = self.backbone(x)
        cls, reg, quad, dx, dy = self.header(features)

        return {"reg_map" : reg, "cls_map":cls, "quad_map": quad, "x": dx, "y": dy}


if __name__ == "__main__":
    geom = {
        "L1": -40.0,
        "L2": 40.0,
        "W1": 0.0,
        "W2": 70.0,
        "H1": -2.5,
        "H2": 1.0,
        "input_shape": [800, 700, 36],
        "label_shape": [200, 175, 7]
    }

    pixor = HotSpotPIXOR(geom)
    print("number of parameters: ", sum(p.numel() for p in pixor.parameters() if p.requires_grad))

    # input = torch.rand((1, 35, 800, 700))
    # pred = pixor(input)
    # print(pred["reg_map"].shape)
    device = "cuda:0"
    pixor.to(device)
    print("cuda ?: ", next(pixor.parameters()).is_cuda)
    num = 30
    total_time = 0
    for i in range(num):
        input = torch.rand((1, 35, 800, 700))
        input = input.to(device)
        start = time.time()
        output = pixor(input)
        print("time it took for one inference: ", time.time() - start)
        total_time += time.time() - start

    print(" time it took for {} inference: {}".format(num, total_time / num))

    deconv = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
    input = torch.rand((1, 196, 50, 43))
    pred = deconv(input)
    print(pred.shape)