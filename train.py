import torch 
import torch.nn as nn
from torch.utils.data import dataset,DataLoader
import torch.optim as optim
from unet import *
from dataset import *
from config import *
from lora import *


betas = torch.linspace(start=1e-4, end=2e-2, steps=T).to(DEVCIE)
alphas = (1 - betas).to(DEVCIE)
alphas_pi = torch.cumprod(alphas, dim=0).to(DEVCIE)
alphas_pi_pre = torch.cat((torch.tensor([1]).to(DEVCIE), alphas_pi[:-1]), dim=0).to(DEVCIE)
variance = ((1 - alphas) * (1 - alphas_pi_pre) / (1 - alphas_pi)).to(DEVCIE)

def forward_diffusion(batch_img_tensor, batch_time):
    batch_size = batch_img_tensor.shape[0]
    img = torch.sqrt(alphas_pi[batch_time].reshape(batch_size, 1, 1, 1)) * batch_img_tensor
    noise = torch.randn_like(batch_img_tensor)
    noise_temp = torch.sqrt((1 - alphas_pi)[batch_time].reshape(batch_size, 1, 1, 1)) * noise
    return img + noise_temp, noise

if __name__ == "__main__":
    # dataset = MNISTDataset(data_dir="mnist_jpg")
    dataset = ZeroTwoDataset(data_dir="02_img_RGB_48")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(len(dataset))
    channel_list = [3, 64, 128, 256, 512, 1024]
    model = Unet(channel_list=channel_list)

    model_path = "ckpt/checkpoints-stable_diffusion_02_RGB_48.pth"
    if os.path.exists(model_path):
        checkpoints = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoints)
        print("模型权重已经被成功加载》》》》》")
    elif USE_LORA:
        torch.save(model.state_dict(), 'checkpoints.pth')
        print("模型初始权重已经被保存！！！！")
        
    if USE_LORA:
        for name, layer in list(model.named_modules()):
            name_clos = name.split('.')
            if (name_clos[-1] in LORA_FINETUNE_LAYERS) and isinstance(layer, nn.Linear):
                  inject_lora(model, name)
        
        lora_path ="lora.pth"
        if os.path.exists(lora_path):
            lora_state = torch.load(lora_path)
            model.load_state_dict(lora_state, strict=False)
            print("lora权重已经被成功加载》》》》》")

        for name, param in model.named_parameters():
            name_clos = name.split('.')
            if not name_clos[-1] in ['lora_a', 'lora_b']:
                param.requires_grad = False
            else:
                param.requires_grad = True 

    model = model.to(DEVCIE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss(reduction='mean')
    for epoch in range(EPOCH):
        for batch_img_tensor, batch_label in dataloader:

            batch_size = batch_img_tensor.shape[0]
            batch_time = torch.randint(0, T, (batch_size, 1))

            batch_img_tensor = batch_img_tensor.to(DEVCIE)
            batch_time = batch_time.to(DEVCIE)
            batch_label = batch_label.to(DEVCIE)

            batch_img_tensor, batch_noise = forward_diffusion(batch_img_tensor, batch_time)
            
            batch_img_tensor = batch_img_tensor.to(DEVCIE)
            batch_noise =batch_noise.to(DEVCIE)

            batch_pred_noise = model(batch_img_tensor, batch_time, batch_label)

            # Compute loss
            loss = mse_loss(batch_pred_noise, batch_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss:{loss}")

        if USE_LORA:
            lora_state = {}
            for name, param in model.named_parameters():
                name_clos = name.split('.')
                if name_clos[-1] in ['lora_a','lora_b']:
                    lora_state[name] = param
            torch.save(lora_state, 'lora.pth')
            print("lora权重已经被保存！！！！")
        else:
            torch.save(model.state_dict(), 'checkpoints.pth')
            print("权重已经被保存！！！！")