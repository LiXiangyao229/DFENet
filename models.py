import torch
import torch.nn as nn
import torch.nn.functional as F
from test import (
    Conv, DFE, EfficientAdditiveAttentions,
    PFC512, PFC256, SFA, Zoom_cat2, CA
)


class Detect(nn.Module):
    def __init__(self, nc=80, channels=()):
        super().__init__()
        self.nc = nc
        self.nl = len(channels)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.conv = nn.Conv2d(channels[0], self.no, 1)

    def forward(self, x):
        return self.conv(x[0])



class DFENet(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        # Backbone
        self.backbone = nn.ModuleList()
        self.backbone.append(Conv(3, 64, 3, 2))  # 0-P1/2
        self.backbone.append(Conv(64, 128, 3, 2))  # 1-P2/4
        self.backbone.append(DFE(128, 128, 3))  # 2
        self.backbone.append(Conv(128, 256, 3, 2))  # 3-P3/8
        self.backbone.append(DFE(256, 256, 6))  # 4
        self.backbone.append(Conv(256, 512, 3, 2))  # 5-P4/16
        self.backbone.append(DFE(512, 512, 6))  # 6
        self.backbone.append(Conv(512, 1024, 3, 2))  # 7-P5/32
        self.backbone.append(DFE(1024, 1024, 3))  # 8
        self.backbone.append(EfficientAdditiveAttentions())  # 9

        # Head
        self.head = nn.ModuleList()
        self.head.append(Conv(1024, 512, 1, 1))  # 10
        self.head.append(Conv(256, 512, 1, 1))  # 11
        self.head.append(PFC512())  # 12

        self.head.append(Conv(512, 256, 1, 1))  # 13
        self.head.append(Conv(128, 256, 1, 1))  # 14
        self.head.append(PFC256())  # 15

        self.head.append(Zoom_cat2())  # 16
        self.head.append(Conv(1024, 1024, 1, 1))  # 17

        self.head.append(Zoom_cat2())  # 18
        self.head.append(Conv(512, 512, 1, 1))  # 19

        self.head.append(Conv(1024, 512, 1, 1))  # 20
        self.head.append(SFA(512, 256))  # 21

        self.head.append(Conv(512, 256, 1, 1))  # 22
        self.head.append(SFA(256, 128))  # 23

        self.head.append(Detect(nc, [256, 512, 1024]))  # 24

        self.channels = [
            64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024,
            512, 512, 1536, 512, 256, 256, 768, 256,
            1024, 1024, 1024,
            512, 512, 512,
            512, 512, 512,
            256, 256, 256,
            nc + 64
        ]

    def forward(self, x):
        outputs = {}

        for i, module in enumerate(self.backbone):
            if i == 0:
                y = module(x)
            elif i == 11:
                y = module(outputs[4])
            else:
                y = module(y)
            outputs[i] = y

        y10 = self.head[0](outputs[9])
        outputs[10] = y10

        y11 = self.head[1](outputs[4])
        outputs[11] = y11

        y12 = self.head[2]([outputs[11], outputs[6], outputs[10]])
        outputs[12] = y12

        y13 = self.head[3](y12)
        outputs[13] = y13

        y14 = self.head[4](y13)
        outputs[14] = y14

        y15 = self.head[5](outputs[2])
        outputs[15] = y15

        y16 = self.head[6]([outputs[15], outputs[4], outputs[14]])
        outputs[16] = y16

        y17 = self.head[7](y16)
        outputs[17] = y17

        y18 = self.head[8]([outputs[13], outputs[10]])
        outputs[18] = y18

        y19 = self.head[9](y18)
        outputs[19] = y19

        y20 = self.head[10](y19)
        outputs[20] = y20

        y21 = self.head[11]([outputs[17], outputs[14]])
        outputs[21] = y21

        y22 = self.head[12](y21)
        outputs[22] = y22

        y23 = self.head[13](y22)
        outputs[23] = y23

        y24 = self.head[14](outputs[20])
        outputs[24] = y24

        y25 = self.head[15](outputs[23], y24)
        outputs[25] = y25

        y26 = self.head[16](y25)
        outputs[26] = y26

        y27 = self.head[17](y26)
        outputs[27] = y27

        y28 = self.head[18](outputs[17], y27)

        y29 = self.head[19](y28)
        outputs[29] = y29

        y30 = self.head[20]([outputs[29], outputs[26], outputs[20]])
        outputs[30] = y30

        return y30


if __name__ == "__main__":

    model = DFENet(nc=300)

    total_params = sum(p.numel() for p in model.parameters())

    x = torch.randn(2, 3, 640, 640)
    out = model(x)