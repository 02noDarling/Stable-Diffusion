import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from time_emb import *

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.time_emb = Time_Embedding(emb_size=EMB_SIZE, channel=out_channel)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, time):
        x = self.conv1(x)
        emb = self.time_emb(time)
        x = x + emb.reshape(emb.shape[0], emb.shape[1], 1, 1)
        return self.conv2(x)

class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel//2),
            nn.ReLU()
        )
    
    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        output = self.conv(up)
        output = torch.cat((output, feature_map), dim=1)
        return output


if __name__ == "__main__":
    conv = ConvBlock(in_channel=3, out_channel=64)
    conv_down = DownSample(channel=3)
    input = torch.randn(10, 3, IMG_SIZE, IMG_SIZE)
    time = torch.randint(0, T, (10, 1))
    output = conv(input, time)
    print(output.shape)