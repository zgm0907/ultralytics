import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels,channels,kernel_size=5,padding=2,groups=channels)
        self.dconv1_7 = nn.Conv2d(channels,channels,kernel_size=(1,7),padding=(0,3),groups=channels)
        self.dconv7_1 = nn.Conv2d(channels,channels,kernel_size=(7,1),padding=(3,0),groups=channels)
        self.dconv1_11 = nn.Conv2d(channels,channels,kernel_size=(1,11),padding=(0,5),groups=channels)
        self.dconv11_1 = nn.Conv2d(channels,channels,kernel_size=(11,1),padding=(5,0),groups=channels)
        self.dconv1_21 = nn.Conv2d(channels,channels,kernel_size=(1,21),padding=(0,10),groups=channels)
        self.dconv21_1 = nn.Conv2d(channels,channels,kernel_size=(21,1),padding=(10,0),groups=channels)
        self.conv = nn.Conv2d(channels,channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
         #在应用通道注意力后直接改变特征，使得后续卷积操作在经过注意力机制的特征图上进行。这样的结构可能更注重通过注意力机制得到的特征，并在此基础上进一步提取多尺度信息
        # inputs = self.ca(inputs)
       
        # 通过通道注意力调整输入特征，然后在经过多个不同尺度的卷积之后，进行特征叠加。该结构更适合在需要强调原始特征和不同尺度特征组合的场景中使用
        channel_att_vec = self.ca(inputs) # 调用通道注意力，得到每一个通道的权值
        inputs = channel_att_vec * inputs # 与输入相乘，得到论文定义的通道先验信息
        

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out
    
    
    
            
        
