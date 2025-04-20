import torch 
import torch.nn as nn
from torch.utils.data import dataset,DataLoader
import torch.optim as optim
from unet import *
from dataset import *
from config import *


betas = torch.linspace(start=1e-4, end=2e-2, steps=T)
alphas = 1 - betas
alphas_pi = torch.cumprod(alphas, dim=0)
alphas_pi_pre = torch.cat((torch.tensor([1]), alphas_pi[:-1]), dim=0)
variance = (1 - alphas) * (1 - alphas_pi_pre) / (1 - alphas_pi)

def forward_diffusion(batch_img_tensor, batch_time):
    batch_size = batch_img_tensor.shape[0]
    img = torch.sqrt(alphas_pi[batch_time].reshape(batch_size, 1, 1, 1)) * batch_img_tensor
    noise = torch.randn_like(batch_img_tensor)
    noise_temp = torch.sqrt((1 - alphas_pi)[batch_time].reshape(batch_size, 1, 1, 1)) * noise
    return img + noise_temp, noise

if __name__ == "__main__":
    dataset = MNISTDataset(data_dir="mnist_jpg")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(len(dataset))
    channel_list = [1, 64, 128, 256, 512, 1024]
    model = Unet(channel_list=channel_list)

    if os.path.exists("checkpoints.pth"):
        checkpoints = torch.load("checkpoints.pth")
        model.load_state_dict(checkpoints)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    mse_loss = nn.MSELoss(reduction='mean')
    for epoch in EPOCH:
        for batch_img_tensor, batch_label in dataloader:
            batch_size = batch_img_tensor.shape[0]
            batch_time = torch.randint(0, T, (batch_size, 1))
            batch_img_tensor, batch_noise = forward_diffusion(batch_img_tensor, batch_time)
            batch_pred_noise = model(batch_img_tensor, batch_time)

            # Compute loss
            loss = mse_loss(batch_pred_noise, batch_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss:{loss}")

        torch.save(model.state_dict(), 'checkpoints.pth')
        print("权重已经被保存！！！！")