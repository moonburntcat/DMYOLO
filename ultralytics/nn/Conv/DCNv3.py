###################### DCNV3  ####     start#############################

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv,autopad
from ultralytics.nn.modules.block import C2f,Bottleneck
from ultralytics.nn.ops_dcnv3.modules import DCNv3


class DCNV3_YOLO(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x


class Bottleneck_DCNV3(Bottleneck):
    """Standard bottleneck with DCNV3."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNV3_YOLO(c_, c2, k[1])



class C2f_DCNv3(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
 
###################### DCNV3  ####     end#############################