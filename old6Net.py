from torch import nn
import torch



class Encoder(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Encoder,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,1,1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,3,1,1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.Downsample = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self,x):
        y = self.conv1(x)
        out = self.Downsample(y)
        return y,out


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(AttentionGate, self).__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels

        # 注意力权重生成器
        self.attention = nn.Sequential(
            nn.ConvTranspose2d(in_channels, gating_channels, 3,2,1,1),
            nn.BatchNorm2d(gating_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gating_channels, gating_channels, kernel_size=1),
            nn.BatchNorm2d(gating_channels),
            nn.Sigmoid()
        )

    def forward(self, x, gating_signal):
        # 生成注意力权重
        attention_weights = self.attention(gating_signal)

        # 特征加权
        attended_features = x * attention_weights

        # 返回加权后的特征
        return attended_features
class Decoder(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Decoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch*2,3,1,1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(out_ch*2,out_ch*2,3,1,1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
        )
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.a1 = AttentionGate(out_ch,out_ch)
    def forward(self,x,concat):
        y = self.conv1(x)
        y = self.Upsample(y)
        weight = self.a1(concat,y)
        out = torch.concat((y,weight),dim=1)
        return out


class old6Net(nn.Module):
    def __init__(self):
        super(old6Net, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=Encoder(3,out_channels[0])#3-64
        self.d2=Encoder(out_channels[0],out_channels[1])#64-128
        self.d3=Encoder(out_channels[1],out_channels[2])#128-256
        #self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        #self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u1=Decoder(out_channels[2],out_channels[2])#256-256
       # self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u2=Decoder(out_channels[3],out_channels[1])#512-256-128
        self.u3=Decoder(out_channels[2],out_channels[0])#256-128-64
        #输出

        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],1,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        #out_4,out4=self.d4(out3)

        out4=self.u1(out3,out_3)

        out5=self.u2(out4,out_2)

        out6=self.u3(out5,out_1)
        #out8=self.u4(out7,out_1)
        out=self.o(out6)
        return out
