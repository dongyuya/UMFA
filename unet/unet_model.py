""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from function import adaptive_instance_normalization
from wct import transform



class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=8, stride=8, return_indices=True)
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=16, stride=16, return_indices=True)
        self.conv = nn.Conv2d(1472, 512, (3, 3), 1)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        # self.in1 = nn.InstanceNorm2d(512)
        self.up2 = Up(512, 256 // factor, bilinear)
        # self.in2 = nn.InstanceNorm2d(256)
        self.up3 = Up(256, 128 // factor, bilinear)
        # self.in3 = nn.InstanceNorm2d(128)
        self.up4 = Up(128, 64, bilinear)
        # self.in4 = nn.InstanceNorm2d(64)
        self.outc = OutConv(64, n_classes)
    def en(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, x4, x3, x2, x1
    # def en_con(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     return x5, x4, x3, x2, x1
    #
    # def en_sty(self, y):
    #     y1 = self.inc(y)
    #     y2 = self.down1(y1)
    #     y3 = self.down2(y2)
    #     y4 = self.down3(y3)
    #     y5 = self.down4(y4)
    #     return y5, y4, y3, y2, y1


    def de1(self, x5, x4):

        x = self.up1(x5, x4)
        return x

    def de2(self, x, x3):

        x = self.up2(x, x3)
        return x

    def de3(self, x, x2):

        x = self.up3(x, x2)
        return x

    def de4(self, x, x1):

        x = self.up4(x, x1)
        return x

    def de5(self, x):
        logits = self.outc(x)
        return logits

    def forward(self, x, y):
        x5, x4, x3, x2, x1 = self.en(x)
        y5, y4, y3, y2, y1 = self.en(y)

        x5 = adaptive_instance_normalization(x5, y5)
        # BFA
        x1_F, _ = self.maxPool_mid1(x1)
        x2_F, _ = self.maxPool_mid2(x2)
        x3_F, _ = self.maxPool_mid3(x3)
        x4_F, _ = self.maxPool_mid4(x4)

        x5 = torch.cat((x5, x4_F), 1)
        x5 = torch.cat((x5, x3_F), 1)
        x5 = torch.cat((x5, x2_F), 1)
        x5 = torch.cat((x5, x1_F), 1)
        x5 = self.conv(x5)

        x4 = adaptive_instance_normalization(x4, y4)
        x3 = adaptive_instance_normalization(x3, y3)
        x2 = adaptive_instance_normalization(x2, y2)
        x1 = adaptive_instance_normalization(x1, y1)

        x = self.de1(x5, x4)
        x = self.de2(x, x3)
        x = self.de3(x, x2)
        x = self.de4(x, x1)
        out = self.de5(x)
        return out


        
        
        
        
    
