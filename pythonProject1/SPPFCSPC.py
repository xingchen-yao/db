import torch
from torch import nn
import torch.nn.functional as F

class SPPFCSPC(nn.Module):
    # 本代码由YOLOAir目标检测交流群 心动 大佬贡献
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(c_, c_, 3, 1)
        self.cv4 = nn.Conv2d(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = nn.Conv2d(4 * c_, c_, 1, 1)
        self.cv6 = nn.Conv2d(c_, c_, 3, 1)
        self.cv7 = nn.Conv2d(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y1 = nn.functional.interpolate(y1,x.shape[2:])
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
if __name__ == '__main__':
    x = torch.randn(3, 256, 640, 640)
    in_channel = x.shape[1]
    model = SPPFCSPC(c1=in_channel,c2=256)
    y = model(x)
    print(model)
    print(y.shape)