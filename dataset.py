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

class ZeroTwoDataset(Dataset):
    def __init__(self,data_dir):
        self.img_paths = []
        self.labels = []

        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(data_dir, filename)
                label = file_path[-5]
                self.img_paths.append(file_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        img_tensor = pil_to_tensor(img)
        label = self.labels[index]
        label = torch.tensor(int(label))

        return img_tensor * 2 -1 , label

def dataset_process(input_dir, output_dir):
    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历input_dir下的所有文件
    for filename in os.listdir(input_dir):
        input_image_path = os.path.join(input_dir, filename)
        
        # 确保只处理图片文件
        if os.path.isfile(input_image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 打开图像文件
            with Image.open(input_image_path) as img:
                # 转换为灰度图
                gray_img = img.convert('RGB')
                
                # Resize 图片到 48x48
                gray_img_resized = gray_img.resize((IMG_SIZE, IMG_SIZE))
                
                # 保存转换后的灰度图像到output_dir，保持原文件名
                output_image_path = os.path.join(output_dir, filename)
                gray_img_resized.save(output_image_path)
                
    print(f"All images have been processed and saved to {output_dir}.")
    
if __name__ == "__main__":
    # image = Image.open(r"D:\02图片\zerotwo.png")
    # image_L = image.convert('RGB')
    # img_tensor = pil_to_tensor(image_L)
    # print(img_tensor)
    # print(img_tensor.shape)

    # reduce_img = tensor_to_pil(img_tensor)
    # reduce_img.show()

    # dataset = MNISTDataset(data_dir="mnist_jpg")
    # img_tensor, label = dataset[0]
    # print(img_tensor.shape)
    # print(label)

    # img = tensor_to_pil(img_tensor)
    # img.show()

    dataset = ZeroTwoDataset(data_dir="02_img")

    for i in range(len(dataset)):
        img_tensor,label = dataset[i]
        img_tensor = (img_tensor + 1) / 2
        img = tensor_to_pil(img_tensor)
        img.show()
    # img_tensor = dataset[4]
    # print(img_tensor.shape)

    # img = tensor_to_pil(img_tensor)
    # img.show()