import torch
from torch import nn
import torch.nn.functional as F
from torchstat import stat


def aspp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self, in_channels, output_stride=16):
        super(ASPP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = aspp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = aspp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = aspp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = aspp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        # initialize_weights(self)


    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

if __name__ == '__main__':
    x = torch.randn(8, 2048, 20, 20)
    in_channel = x.shape[1]
    aspp = ASPP(in_channels=in_channel)
    y = aspp(x)
    print(aspp)
    print(y.shape)
    stat(aspp, input_size=[2048, 20, 20])
    total = sum([param.nelement() for param in aspp.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))

    # torch.Size([3, 256, 640, 640])