import torch
import torch.nn as nn



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ECA(nn.Module):
 

    def __init__(self, channel, k_size=5):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class MSFE(nn.Module):

    def __init__(self, c_in, c_out):
        super(MSFE, self).__init__()

        c_ = int(c_in//4)

        self.ECA = ECA(c_in, k_size=5)
        self.branch1 = Conv(c_in, c_, 1, 1)
        self.branch2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch3 = nn.Sequential(
            Conv(c_in, c_, (3, 1), s=1, p=(1, 0)),
            Conv(c_, c_, (1, 3), s=1, p=(0, 1))
        )
        self.branch4 = nn.Sequential(
            Conv(c_in, c_, (3, 1), s=1, p=(1, 0)),
            Conv(c_, c_, (1, 3), s=1, p=(0, 1)),
            Conv(c_, c_, (3, 1), s=1, p=(1, 0)),
            Conv(c_, c_, (1, 3), s=1, p=(0, 1))
        )
        self.branch5 = nn.Sequential(
            Conv(c_in, c_, (5, 1), s=1, p=(2, 0)),
            Conv(c_, c_, (1, 5), s=1, p=(0, 2)),
            Conv(c_, c_, (5, 1), s=1, p=(2, 0)),
            Conv(c_, c_, (1, 5), s=1, p=(0, 2))
        )
        self.conv = Conv(c_in*2, c_out, k=1)
    def forward(self, x):
        x1 = self.ECA(x)
        y1 = self.branch1(x1)
        y2 = self.branch2(x1)
        y3 = self.branch3(x1)
        y4 = self.branch4(x1)
        y5 = self.branch5(x1)
        out = x + self.conv(torch.cat([y1, y2, y3, y4, y5], 1))
        return out
