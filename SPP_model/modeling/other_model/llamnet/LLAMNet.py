from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.lib.resnet_module import Stem, ResLayer
# from models.lib.DMAM import DMAM,  ConvBnReLU
from .LCAM import LCAM
from .MFM import MFM



class LLAMNet(nn.Module):
    def __init__(self, classes = 1):
        super(LLAMNet, self).__init__()

        n_blocks = [3, 4, 6, 3]    # backbone=resnet101,num_eachlayer_bottleneck=[3,4,23,3]
        multi_grids = [1, 2, 4]    # 3x3Conv in layer5 each dilation is dilation*[1,2,4]

        self.layer1 = Stem(64)   # 下采样X4
        self.layer2 = ResLayer(n_blocks[0], 64, 256, stride=1, dialtion=1)
        self.layer3 = ResLayer(n_blocks[1], 256, 512, stride=2, dialtion=1)    #
        self.layer4 = ResLayer(n_blocks[2], 512, 1024, stride=1, dialtion=2)   #
        self.layer5 = ResLayer(n_blocks[3], 1024, 2048, stride=1, dialtion=4, multi_grids=multi_grids)

        atrous_rates = [6, 12, 18]  # 3x3Conv in ASPP each dilation is 6,12,18

        self.aspp1 = DMAM(2048, 256, atrous_rates)
        self.fc1 = ConvBnReLU((len(atrous_rates)+3)*256,256,kernel_size=1,stride=1,padding=0)

        # Decoder
        self.reduce1 = ConvBnReLU(256, 48, kernel_size=1, stride=1, padding=0)
        self.reduce2 = ConvBnReLU(512, 96, kernel_size=1, stride=1, padding=0)
        self.conv1_1 = ConvBnReLU(96, 48, kernel_size=1, stride=1, padding=0)


        #self.atten1 = HAttention(64, 64)
        self.atten2 = LCAM(256, 256)
        self.atten1 = LCAM(512, 512)
        self.atten3 = LCAM(1536, 1536)
        self.bga = MFM(classes)



    def forward(self, x):
        h1 = self.layer1(x)   #64*64*64

        h2 = self.layer2(h1)   #256*64*64
        h2 = self.atten2(h2)

        h_1 = self.reduce1(h2)

        h3 = self.layer3(h2)   #512*32*32
        h3 = self.atten1(h3)

        h_2 = self.reduce2(h3) #96*32*32
        h_2 = self.conv1_1(h_2) #48*32*32

        h4 = self.layer4(h3)   #1024*32*32
        h5 = self.layer5(h4)   #2048*32*32
        h6 = self.aspp1(h5)     #1536*32*32
        h6 = self.atten3(h6)
        h7 = self.fc1(h6)

        h8 = F.interpolate(h7, size=h_1.shape[2:], mode="bilinear", align_corners=False)
        h9 = torch.cat((h_1, h8), dim=1)

        h13 = torch.cat((h_2, h7), dim=1)

        h14 = self.bga(h9, h13)

        h11 = F.interpolate(h14, size=x.shape[2:], mode="bilinear", align_corners=False)
        return h11



class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, downsample):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        self.conv3X3_1 = ConvBnReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
                                  dilation=dilation, relu=True)
        self.conv3X3_2 = ConvBnReLU(mid_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, relu=True)

        self.shortcut =ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=False)

    def forward(self, x):

        x_ = self.conv3X3_1(x)

        x_ = self.conv3X3_2(x_)

        print('==================================')
        print(x_.shape)
        print(x.shape)

        if self.downsample:
            x_ += self.shortcut(x)
            print('!!!!!!!!!!!!!!!!!!!!!!')
            print(x_.shape)
        else:
            x_ += x
        return F.relu(x_)

class Basicneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, downsample):
        super(Basicneck, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4
        self.reduce = ConvBnReLU(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=True)
        self.conv3X3 = ConvBnReLU(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, relu=True)
        self.increase = ConvBnReLU(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, relu=False)
        self.shortcut = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=False)

    def forward(self, x):
        x_ = self.reduce(x)
        x_ = self.conv3X3(x_)
        x_ = self.increase(x_)
        if self.downsample:
            x_ += self.shortcut(x)
        else:
            x_ += x
        return F.relu(x_)



class ResLayer(nn.Sequential):
    def __init__(self, num_layers, in_channels, out_channels, stride, dialtion, multi_grids=None):
        super(ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(num_layers)]
        else:
            assert num_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(num_layers):
            self.add_module(
                'block{}'.format(i+1), Basicneck(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    stride=(stride if i == 0 else 1),
                    dilation=dialtion * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


"""
The first conv layer
Note that the max pooling is different from both MSRA and FAIR ResNet.
"""


class Stem(nn.Sequential):
    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.add_module('conv1', ConvBnReLU(3, out_channels, kernel_size=7, stride=2, padding=3, dilation=1))
        self.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))




# class ConvBnReLU(nn.Sequential):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
#         super(ConvBnReLU, self).__init__()
#         self.add_module(
#             'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
#         )
#         self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
#         if relu:
#             self.add_module('ReLU', nn.ReLU())
class ConvBnReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module('ReLU', nn.ReLU())

class DWConvBnReLU(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation=1, groups=1, relu=True):
        super(DWConvBnReLU, self).__init__()
        self.add_module(
            'DWConv', nn.Conv2d(in_channels, in_channels, kernel_size1, stride, padding1, dilation, groups, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.999)
        )
        self.add_module('Conv',nn.Conv2d(in_channels,out_channels, kernel_size2, stride, padding2, bias=False)
        )
        if relu:
            self.add_module('ReLU', nn.ReLU()
        )



class ImagePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.conv = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        x_size = x.shape
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x


class DMAM(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(DMAM, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module('c0', ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1))   # channel->out_channels, keep size

        for idx, rate in enumerate(rates):
            self.stages.add_module('c{}'.format(idx+1), DWConvBnReLU(in_channels, out_channels, kernel_size1=3, kernel_size2=1, stride=1, padding1=rate, groups=in_channels, padding2=0, dilation=rate))   # channel->out_channels, keep size
        self.stages.add_module('imagepool', ImagePool(in_channels, out_channels))    # channel->out_channels, keep size
        self.stages.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):

        x = torch.cat([stage(x) for stage in self.stages.children()], dim=1)

        return x