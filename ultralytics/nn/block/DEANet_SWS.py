import torch
from torch import nn
from einops import rearrange
 
from ultralytics.nn.modules.conv import Conv, autopad
 
 
class SimAMWithSlicing(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAMWithSlicing, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
 
        block_size_h = height // 2
        block_size_w = width // 2
 
        block1 = x[:, :, :block_size_h, :block_size_w]
        block2 = x[:, :, :block_size_h, block_size_w:]
        block3 = x[:, :, block_size_h:, :block_size_w]
        block4 = x[:, :, block_size_h:, block_size_w:]
 
        enhanced_blocks = []
        for block in [block1, block2, block3, block4]:
            n = block_size_h * block_size_w - 1
            block_minus_mu_square = (block - block.mean(dim=[2, 3], keepdim=True)).pow(2)
            y = block_minus_mu_square / (
                    4 * (block_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
            enhanced_blocks.append(block * self.activation(y))
 
        enhanced_image = torch.cat([torch.cat([enhanced_blocks[0], enhanced_blocks[1]], dim=3),
                                    torch.cat([enhanced_blocks[2], enhanced_blocks[3]], dim=3)], dim=2)
 
        return enhanced_image
 
 
class Conv_SWS(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_SWS, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SimAMWithSlicing(c2)
 
    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))
 
    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))
 
 
 
class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)
 
    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn
 
 
class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
 
    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn
 
    
class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
 
class CGAFusion_SWS(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion_SWS, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.SWS =Conv_SWS(dim,dim)
 
    def forward(self, data):
        x, y = data
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.SWS(result)
        #result = self.conv(result)
        return result
