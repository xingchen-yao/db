import torch
from torch import nn

class channel_attention(nn.Module):
    def __init__(self,channel,ratio=16):
        super(channel_attention,self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel//ratio,channel,False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b,c])
        avg_pool_out = self.avg_pool(x).view([b,c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b,c,1,1])
        return out * x


class spatial_attention(nn.Module):
    def __init__(self,kernel_size=7):
        super(spatial_attention,self).__init__()
        padding = 7//2
        self.conv = nn.Conv2d(2,1,kernel_size,padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        max_pool_out = torch.max(x,dim=1,keepdim=True)
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out,avg_pool_out],dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out*x

class Cbam(nn.Module):
    def __init__(self, channel, ratio=16,kernel_size=7):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(channel,ratio)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self,x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

model = Cbam(512)
print(model)
inputs = torch.ones([2,512,26,26])
outputs = model(inputs)