import torch
import torch.nn as nn
from conv_block import *
from config import *

class Unet(nn.Module):
    def __init__(self, channel_list=[3, 64, 128, 256, 512, 1024]):
        super(Unet, self).__init__()

        self.channel_list = channel_list

        self.cls_embedding = nn.Embedding(10, EMB_SIZE)

        self.enc_blocks = nn.ModuleList()
        for i in range(1, len(channel_list)):
            self.enc_blocks.append(ConvBlock(in_channel=channel_list[i-1], out_channel=channel_list[i]))
        
        self.down_blocks = nn.ModuleList()
        for i in range(1, len(channel_list)-1):
            self.down_blocks.append(DownSample(channel=channel_list[i]))
        
        self.up_blocks = nn.ModuleList()
        for i in range(len(channel_list)-1, 1, -1):
            self.up_blocks.append(UpSample(channel=channel_list[i]))

        self.dec_blocks = nn.ModuleList()
        for i in range(len(channel_list)-1, 1, -1):
            self.dec_blocks.append(ConvBlock(in_channel=channel_list[i], out_channel=channel_list[i-1]))

        self.conv = nn.Conv2d(in_channels=channel_list[1], out_channels=channel_list[0], kernel_size=3, stride=1, padding=1)

    def forward(self, x, time, cls):
        cls = self.cls_embedding(cls)
        residual = []
        for i in range(len(self.enc_blocks)-1):
            x = self.enc_blocks[i](x, time, cls)
            residual.append(x)
            x = self.down_blocks[i](x)
        
        x = self.enc_blocks[-1](x, time, cls)

        residual.reverse()

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, residual[i])
            x = self.dec_blocks[i](x, time, cls)
        
        x = self.conv(x)
        return x

if __name__ == "__main__":
    input = torch.randn(10, 1, IMG_SIZE, IMG_SIZE)
    time = torch.randint(0, 10, (10,1))
    cls = torch.randint(0, 10, (10,))
    channel_list = [1, 64, 128, 256, 512, 1024]
    model = Unet(channel_list=channel_list)
    output = model(input, time, cls)
    print(output.shape)
