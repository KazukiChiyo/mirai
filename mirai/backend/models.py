# Author: Kexuan Zou
# Date: Feb 4, 2019
# License: MIT

import torch.nn as nn
from .layers import Conv2DBatchNorm, UNetConv2, UNetUpsample

# reference: https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def n_trainables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        raise NotImplementedError


# reference: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py
class UNet(BaseModel):
    def __init__(self, in_channels, out_channels, feature_scale=1, deconv=True, batch_norm=True):
        super(UNet, self).__init__()
        self.deconv = deconv
        self.in_channels = in_channels
        self.batch_norm = batch_norm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = UNetConv2(self.in_channels, filters[0], kernel_initializer='kaiming', batch_norm=self.batch_norm)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = UNetConv2(filters[0], filters[1], kernel_initializer='kaiming', batch_norm=self.batch_norm)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = UNetConv2(filters[1], filters[2], kernel_initializer='kaiming', batch_norm=self.batch_norm)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = UNetConv2(filters[2], filters[3], kernel_initializer='kaiming', batch_norm=self.batch_norm)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = UNetConv2(filters[3], filters[3], kernel_initializer='kaiming', batch_norm=self.batch_norm)

        self.up_concat4 = UNetUpsample(filters[4], filters[2], self.deconv)
        self.up_concat3 = UNetUpsample(filters[3], filters[1], self.deconv)
        self.up_concat2 = UNetUpsample(filters[2], filters[0], self.deconv)
        self.up_concat1 = UNetUpsample(filters[1], filters[0], self.deconv)

        self.final = Conv2DBatchNorm(filters[0], out_channels, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)

        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)
        out = self.final(up1)

        return out
