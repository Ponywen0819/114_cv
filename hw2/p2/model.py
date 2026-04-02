# ============================================================================
# File: model.py
# Date: 2026-03-27
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # Encoder (contracting path)
        self.enc1 = DoubleConv(3, 64)       # (B, 64, 32, 32)
        self.pool1 = nn.MaxPool2d(2)         # -> (B, 64, 16, 16)
        self.enc2 = DoubleConv(64, 128)      # (B, 128, 16, 16)
        self.pool2 = nn.MaxPool2d(2)         # -> (B, 128, 8, 8)
        self.enc3 = DoubleConv(128, 256)     # (B, 256, 8, 8)
        self.pool3 = nn.MaxPool2d(2)         # -> (B, 256, 4, 4)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)  # (B, 512, 4, 4)

        # Decoder (expansive path)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # -> (B, 256, 8, 8)
        self.dec3 = DoubleConv(512, 256)     # cat with enc3: 256+256=512

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # -> (B, 128, 16, 16)
        self.dec2 = DoubleConv(256, 128)     # cat with enc2: 128+128=256

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # -> (B, 64, 32, 32)
        self.dec1 = DoubleConv(128, 64)      # cat with enc1: 64+64=128

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                       # (B, 64, 32, 32)
        e2 = self.enc2(self.pool1(e1))           # (B, 128, 16, 16)
        e3 = self.enc3(self.pool2(e2))           # (B, 256, 8, 8)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))      # (B, 512, 4, 4)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, 256, 8, 8)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128, 16, 16)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 64, 32, 32)

        # Classification
        out = self.global_avg_pool(d1).flatten(1)  # (B, 64)
        return self.fc(out)                         # (B, 10)
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        # try to load the pretrained weights
        self.resnet = models.resnet18(weights=None)  # Python3.8 w/ torch 2.2.1
        # self.resnet = models.resnet18(pretrained=False)  # Python3.6 w/ torch 1.10.1
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optional):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################

        ############################## TODO End ###############################

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
