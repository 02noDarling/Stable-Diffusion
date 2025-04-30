import torch
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
from config import *

# PIL图像转tensor
pil_to_tensor=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),    # PIL图像尺寸统一  
    transforms.ToTensor()                       # PIL图像转tensor, (H,W,C) ->（C,H,W）,像素值[0,1]
])

tensor_to_pil = transforms.Compose([
    transforms.ToPILImage()
])
class MNISTDataset(Dataset):
    def __init__(self, data_dir):
        self.img_paths = []
        self.labels = []

        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg"):
                file_path = os.path.join(data_dir, filename)
                label = file_path[-5]
                self.img_paths.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("L")
        img_tensor = pil_to_tensor(img)
        label = self.labels[index]
        label = torch.tensor(int(label))

        return img_tensor * 2 -1, label

if __name__ == "__main__":
    # image = Image.open(r"D:\02图片\zerotwo.png")
    # image_L = image.convert('RGB')
    # img_tensor = pil_to_tensor(image_L)
    # print(img_tensor)
    # print(img_tensor.shape)

    # reduce_img = tensor_to_pil(img_tensor)
    # reduce_img.show()

    dataset = MNISTDataset(data_dir="mnist_jpg")
    img_tensor, label = dataset[0]
    print(img_tensor.shape)
    print(label)

    img = tensor_to_pil(img_tensor)
    img.show()