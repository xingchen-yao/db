import torch
import torch.nn as nn

class RRGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.FNUM = 3  # 迭代次数
        self.SepareConv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )
        channels2 = in_channels*2
        self.SepareConv1 = nn.Sequential(
            nn.Conv2d(channels2, channels2, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(channels2, channels2, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(channels2, in_channels, kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x):

        f = self.SepareConv0(x) # channel=256
        b1 = torch.cat([x, f], dim=1) # channel =512
        f1 = self.SepareConv1(b1) # 256
        f2 = torch.cat([f,f1], dim=1)
        f3= self.SepareConv1(f2)
        return f3


if __name__ == "__main__":
    a = torch.rand(1, 256, 160, 160)
    model = RRGN(in_channels=256)
    y = model(a)
    print(y.shape)
