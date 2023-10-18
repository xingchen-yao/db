# -----------------------------------------#
#   SP条形池化模块，输入通道=输出通道=256
# -----------------------------------------#
import torch
from torch import nn
import torch.nn.functional as F
from torchstat import stat


class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels,inter_channels,1,dilation=1,bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, dilation=6, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, dilation=12, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, dilation=18, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv3_1 = nn.Sequential(nn.Conv2d(inter_channels,inter_channels,1,dilation=1,bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, dilation=6, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, dilation=12, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv3_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, dilation=18, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        # self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
        #                            nn.BatchNorm2d(inter_channels),
        #                            nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels*5, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(True))
        self.sigmoid = nn.Sigmoid()

        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)  # 1*1卷积 压缩通道4倍
        xp1 = self.pool1(x1)
        x2 = self.conv2(xp1)
        x3 = self.conv2_1(xp1)
        x4 = self.conv2_2(xp1)
        x5 = self.conv2_3(xp1)
        x6 = self.conv2_4(xp1)

        x7 = F.interpolate(torch.cat((x2,x3,x4,x5,x6),dim=1), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        xp2 = self.pool2(x1)
        x2_1 = self.conv2(xp2)
        x3_1 = self.conv3_1(xp2)
        x4_1 = self.conv3_2(xp2)
        x5_1 = self.conv3_3(xp2)
        x6_1 = self.conv3_4(xp2)

        x8 = F.interpolate(torch.cat((x2_1,x3_1,x4_1,x5_1,x6_1),dim=1), (h, w), **self._up_kwargs)  # 结构图的H*1的部分

        x9 = self.conv4((x7 + x8)) # 结合1*W和H*1的特征
        out = self.sigmoid(x9)
        out = x*out

        return out  # 将输出的特征与原始输入特征结合

if __name__ == "__main__":
    # 构造输入层 [b,c,h,w]==[4,32,16,16]
    inputs = torch.rand([8, 2048, 20, 20])

    # 模型实例化
    model = StripPooling(in_channels=2048)
    # 前向传播
    outputs = model(inputs)

    print(outputs.shape)  # 查看输出结果
    print(model)  # 查看网络结构
    stat(model, input_size=[2048, 20, 20])  # 查看网络参数