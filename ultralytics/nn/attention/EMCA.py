import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

from ultralytics.nn.modules.conv import Conv

class EMCA(nn.Module):

    def __init__(self, groups, kernel_size=3):
        super().__init__()
        self.groups = groups
        self.init_weights()

        self.gap=nn.AdaptiveAvgPool2d(1)
        self.gap_max=nn.AdaptiveMaxPool2d(1)
        self.conv=nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)
        y1=self.gap(x)
        y2=self.gap_max(x)
        y=y1+y2
        y=y.squeeze(-1).permute(0,2,1)
        y=self.conv(y)
        y=self.sigmoid(y)
        y=y.permute(0,2,1).unsqueeze(-1)
        out=x*y.expand_as(x)
        out = out.view(b, c, h, w)
        return out

class PSEMCA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert (c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = EMCA(self.c)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))