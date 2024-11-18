import torch
import torch.nn as nn
from .block import C2f, C3, Bottleneck
from .conv import Conv
 
# Squeeze-and-Excitation (SE) 层，用于通过通道加权来重新校准特征图
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
 
        # 自适应平均池化，将每个通道缩小到单个值（1x1 的空间大小）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # SE块的全连接层，包含一个用于控制复杂度的降维率
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 减少通道维度
            nn.ReLU(inplace=True),  # ReLU激活函数引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 恢复原始通道维度
            nn.Sigmoid()  # Sigmoid激活，将每个通道的权重限制在0到1之间
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入张量的批量大小和通道数量
        y = self.avg_pool(x).view(b, c)  # 对每个通道进行全局平均池化并展平
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层生成每个通道的权重
        return x * y.expand_as(x)  # 对输入特征图进行通道加权
 
 
# 频谱动态聚合层
class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()
 
        # 用于处理幅度部分的卷积和激活操作
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积保持特征图大小不变
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活函数
            SELayer(channel=nc),  # 加入SE层进行通道加权
            nn.Conv2d(nc, nc, 1, 1, 0))  # 另一个1x1卷积
 
        # 用于处理相位部分的卷积和激活操作
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积保持特征图大小不变
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活函数
            SELayer(channel=nc),  # 加入SE层进行通道加权
            nn.Conv2d(nc, nc, 1, 1, 0))  # 另一个1x1卷积
 
    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
 
        # 获取输入张量的幅度和相位信息
        ori_mag = torch.abs(x_freq)  # 计算复数张量的幅度
        ori_pha = torch.angle(x_freq)  # 计算复数张量的相位
 
        # 处理幅度信息
        mag = self.processmag(ori_mag)  # 使用处理幅度的网络
        mag = ori_mag + mag  # 将处理后的结果与原始幅度相加
 
        # 处理相位信息
        pha = self.processpha(ori_pha)  # 使用处理相位的网络
        pha = ori_pha + pha  # 将处理后的结果与原始相位相加
 
        # 重建复数形式的输出
        real = mag * torch.cos(pha)  # 实部：幅度 * cos(相位)
        imag = mag * torch.sin(pha)  # 虚部：幅度 * sin(相位)
        x_out = torch.complex(real, imag)  # 组合成复数输出
 
        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
 
 
        return x_freq_spatial  # 返回处理后的复数张量
 
 
class Bottleneck_FSDA(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Frequency_Spectrum_Dynamic_Aggregation(c_)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
 
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_FSDA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
 
# 在c3k=True时，使用C3k2_PCFN特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_FSDA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
 
 
if __name__ == '__main__':
    FSDA = Frequency_Spectrum_Dynamic_Aggregation(256)
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =FSDA(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)
