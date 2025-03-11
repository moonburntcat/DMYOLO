
import torch
from torch import nn
import torch.nn.functional as F




class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        #self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans