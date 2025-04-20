import torch

# model
IMG_SIZE = 48
T = 1000
EMB_SIZE = 256

#train
EPOCH = 1000
BATCH_SIZE = 200
DEVCIE = 'cuda' if torch.cuda.is_available() else 'cpu'