import torch
import torch.nn as nn
from config import *
from unet import *
from dataset import *
from train import *
from lora import *
import matplotlib.pyplot as plt

channel_list = [3, 64, 128, 256, 512, 1024]
model = Unet(channel_list=channel_list)
model_path = "ckpt/checkpoints-stable_diffusion_02_RGB_48.pth"
checkpoints = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoints)
# for name, layer in model.named_parameters():
#     print(name)
    # print(layer)
# print(type(model.state_dict()))
# for key, value in model.state_dict().items():
#     print(key) 

if USE_LORA:
    for name, layer in list(model.named_modules()):
        name_clos = name.split('.')
        if (name_clos[-1] in LORA_FINETUNE_LAYERS) and isinstance(layer, nn.Linear):
                inject_lora(model, name) 

    lora_state = torch.load('lora.pth')
    model.load_state_dict(lora_state, strict=False)

    for name, layer in list(model.named_modules()):
        if isinstance(layer, LoraLayer):
            layer.raw_linear.weight = nn.Parameter(layer.raw_linear.weight.add(torch.matmul(layer.lora_a, layer.lora_b).T * LORA_ALPHA / LORA_R))
            cur_layer = model
            name_cols = name.split('.')
            for item in name_cols[:-1]:
                cur_layer = getattr(cur_layer, item)
            setattr(cur_layer, name_cols[-1], layer.raw_linear)

model = model.to(DEVCIE)

def backword_denoise(batch_img, batch_cls):
    model.eval()
    batch_size = batch_img.shape[0]
    with torch.no_grad():
        for i in range(T-1, -1, -1):
            batch_time = torch.tensor([[i] for j in range(batch_size)])
            noise = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE).to(DEVCIE)
            batch_img = batch_img.to(DEVCIE)
            batch_time =batch_time.to(DEVCIE)
            batch_cls =batch_cls.to(DEVCIE)
            batch_pred_noise = model(batch_img, batch_time, batch_cls)
            if i!=0 :
                batch_img = 1 / torch.sqrt(alphas[i]) * (batch_img - (1 - alphas[i])/torch.sqrt(1 - alphas_pi[i]) * batch_pred_noise) + torch.sqrt(variance[i]) * noise
            else:
                batch_img = 1 / torch.sqrt(alphas[i]) * (batch_img - (1 - alphas[i])/torch.sqrt(1 - alphas_pi[i]) * batch_pred_noise)
            batch_img = torch.clamp(batch_img, min=-1, max=1)
    return batch_img

if __name__ == "__main__":
    batch_size = 3
    channel = 3
    batch_img = torch.randn(batch_size, channel, IMG_SIZE, IMG_SIZE)
    batch_cls = torch.arange(0, batch_size)
    batch_img = backword_denoise(batch_img, batch_cls)
    batch_img = (batch_img + 1) / 2
    # batch_img =tensor_to_pil(batch_img)
    # print(batch_img) 

    fig, axes = plt.subplots(batch_size, 1, figsize=(5, batch_size * 3))
    
    # If batch_size is 1, axes is not a list, so convert to list for consistency
    if batch_size == 1:
        axes = [axes]
    
    # Convert each image to PIL and display
    for i in range(batch_size):
        # Extract single image: [1, image_size, image_size]
        img_tensor = batch_img[i]  # Shape: [1, image_size, image_size]
        
        # Convert to PIL Image
        pil_img = tensor_to_pil(img_tensor)
        
        # pil_img.save(f"img/output_{i+1}.jpg")
        
        # Display image
        axes[i].imshow(pil_img, cmap='gray')
        axes[i].axis('off')  # Hide axes
        axes[i].set_title(f'Image {i+1}')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
