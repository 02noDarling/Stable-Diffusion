import torch
import torch.nn as nn
import math
from config import *

class Time_Embedding(nn.Module):
    def __init__(self, emb_size, channel):
        super(Time_Embedding, self).__init__()
        self.emb_size = emb_size
        self.linear = nn.Linear(in_features=emb_size, out_features=channel)
    
    def forward(self, time):
        batch_size = time.shape[0]
        position = torch.exp(torch.arange(0, self.emb_size, 2) * (math.log(10000)/self.emb_size))
        position_encoding = torch.zeros(batch_size, self.emb_size)
        position_encoding[:,0::2] = torch.sin(time * position)
        position_encoding[:,1::2] = torch.cos(time * position)
        position_encoding = self.linear(position_encoding)
        return position_encoding

if __name__ == "__main__":
    time = torch.randint(0,5, (10,1))
    print(time)
    time_emb = Time_Embedding(emb_size=EMB_SIZE, channel=16)
    emb = time_emb(time)
    print(emb.shape)