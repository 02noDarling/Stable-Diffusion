import torch
import torch.nn as nn
from unet import *
from config import *
from lora import *

def merge(model_path, lora_pth, output_path):
    channel_list = [3, 64, 128, 256, 512, 1024]
    model = Unet(channel_list=channel_list)
    checkpoints = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoints)

    for name, layer in list(model.named_modules()):
        name_clos = name.split('.')
        if (name_clos[-1] in LORA_FINETUNE_LAYERS) and isinstance(layer, nn.Linear):
                inject_lora(model, name) 

    lora_state = torch.load(lora_pth)
    model.load_state_dict(lora_state, strict=False)

    for name, layer in list(model.named_modules()):
        if isinstance(layer, LoraLayer):
            layer.raw_linear.weight = nn.Parameter(layer.raw_linear.weight.add(torch.matmul(layer.lora_a, layer.lora_b).T * LORA_ALPHA / LORA_R))
            cur_layer = model
            name_cols = name.split('.')
            for item in name_cols[:-1]:
                cur_layer = getattr(cur_layer, item)
            setattr(cur_layer, name_cols[-1], layer.raw_linear)
    torch.save(model.state_dict(), output_path)
    print("权重已经被保存！！！！")

if __name__ == "__main__":
    model_path = "ckpt/checkpoints-stable_diffusion_02_RGB_48.pth"
    lora_path = "lora.pth"
    output_path = "checkpoints.pth"
    merge(model_path, lora_path, output_path)
