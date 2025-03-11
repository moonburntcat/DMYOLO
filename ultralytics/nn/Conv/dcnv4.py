import torch
import torch.nn as nn
import torch.nn.functional as F
 
from ultralytics.nn.modules import (Conv,C3, Bottleneck,C2f)
 
 
try:
    from DCNv4.modules.dcnv4 import DCNv4
except ImportError as e:
    pass
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
class DCNV4_YOLO(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        x = self.dcnv4(x, (x.size(2), x.size(3)))
        x = self.act(self.bn(x))
        return x
 
class Bottleneck_DCNV4(Bottleneck):
    """Standard bottleneck with DCNV3."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNV4_YOLO(c_, c2, k[1])
 
class C3_DCNv4(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV4(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))
 
class C2f_DCNv4(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV4(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))