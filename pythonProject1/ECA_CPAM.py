import torch
from torch import nn
import math
from torchstat import stat  # 查看网络参数


# 定义ECANet的类
class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()

        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size+1

        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs

class CPAM(nn.Module):

    def __init__(self,in_channel=256):
        super(CPAM, self).__init__()


        # 横卷
        self.conv1h = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 1), padding=(1, 0), stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(True))
        # 竖卷
        self.conv1s = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3), padding=(0, 1), stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(True))
        self.conv222 = nn.Sequential(nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel * 2, kernel_size=1, padding=0, stride=1),
                                     nn.BatchNorm2d(in_channel * 2),
                                     nn.ReLU(True))
        self.convout = nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        n, c, h, w = x.size()


        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max,_ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)],dim=1))

        out = self.convout(yap)

        return out

class ECA_CPAM(nn.Module):
    def __init__(self, in_channel):
        super(ECA_CPAM, self).__init__()
        self.eca = eca_block(in_channel)
        self.cpam = CPAM(in_channel)

    def forward(self,x):
        x = self.eca(x)
        x = self.cpam(x)
        return x

if __name__ == "__main__":
    # 构造输入层 [b,c,h,w]==[4,32,16,16]
    inputs = torch.rand([8, 256, 160, 160])
    # 获取输入图像的通道数
    in_channel = inputs.shape[1]
    # 模型实例化
    model = ECA_CPAM(in_channel=in_channel)
    # 前向传播
    outputs = model(inputs)

    print(outputs.shape)  # 查看输出结果
    print(model)  # 查看网络结构
    stat(model, input_size=[256, 160, 160])  # 查看网络参数