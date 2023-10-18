
from torch.nn import functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)

import torch
import torch.nn as nn



class CDAM2(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=7,in_channel=256):
        super(CDAM2, self).__init__()


        self.relu1 = nn.ReLU()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(None, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(None, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(in_channel* 4, in_channel, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=in_channel*2, out_channels=in_channel*2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=in_channel*2, out_channels=in_channel*2, kernel_size=1, padding=0, stride=1)

        # 横卷
        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(None, 1), padding=(0, 0), stride=1)
        # 竖卷
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, None), padding=(0, 0), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        n, c, h, w = x.size()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        y1 = self.avg_pool_x(x)
        y1 = y1.reshape(n, c, h)
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        y2 = self.avg_pool_y(x)
        y2 = y2.reshape(n, c, w)

        # Two different branches of ECA module
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        yac = self.conv111(torch.cat([x * y1.expand_as(x), x * y2.expand_as(x)],dim=1))

        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max,_ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)],dim=1))

        out = self.convout(torch.cat([yac, yap], dim=1))

        return out
if __name__ == "__main__":
    a = torch.rand(1, 256, 40, 50)
    cdam = CDAM2()
    y = cdam(a)
    print(y.shape)