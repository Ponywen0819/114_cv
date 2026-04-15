# ============================================================================
# File: model.py
# Date: 2026-03-27
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch.nn as nn
import torchvision.models as models


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.md = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )  # (B, 1280, 1, 1)
        self.md.classifier[3] = nn.Linear(
            self.md.classifier[3].in_features, 10
        )  # (B, 10)

        self.md.features[0][0].stride = (
            1,
            1,
        )  # change the stride of the first convolution layer

    def forward(self, x):
        out = self.md(x)  # (B, 10)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

        #######################################################################
        # TODO (optional):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. #
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # self.resnet.layer2[0].conv1.stride = (1, 1)
        # self.resnet.layer2[0].downsample[0].stride = (1, 1)
        # self.resnet.layer3[0].conv1.stride = (1, 1)
        # self.resnet.layer3[0].downsample[0].stride = (1, 1)
        self.resnet.layer4[0].conv1.stride = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)

        self.resnet.maxpool = nn.Identity()

        ############################## TODO End ###############################

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":
    model = ResNet18()
    print(model)
    print(sum(p.numel() for p in model.parameters()) / 10**6, "M parameters")
    model = MyNet()
    print(model)
    print(sum(p.numel() for p in model.parameters()) / 10**6, "M parameters")
