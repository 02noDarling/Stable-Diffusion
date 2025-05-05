import torch
import torch.nn as nn
import copy
import math
from config import *

class LoraLayer(nn.Module):
    def __init__(self, raw_linear, in_features, out_features):
        super(LoraLayer, self).__init__()
        self.raw_linear = raw_linear
        self.in_features = in_features
        self.out_features = out_features

        self.lora_a = nn.Parameter(torch.empty(in_features, LORA_R))
        self.lora_b = nn.Parameter(torch.zeros(LORA_R, out_features))
        
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        x = self.raw_linear(x) + torch.matmul(x, torch.matmul(self.lora_a, self.lora_b)) * LORA_ALPHA / LORA_R
        return x

def inject_lora(model, name):
    name_clos = name.split('.')
    cur_layer = model
    for item in name_clos[:-1]:
        cur_layer = getattr(cur_layer, item)
    layer = getattr(cur_layer, name_clos[-1])
    setattr(cur_layer, name_clos[-1], LoraLayer(layer, layer.in_features, layer.out_features))