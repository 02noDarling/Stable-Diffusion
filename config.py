import torch

# model
IMG_SIZE = 48
T = 1000
EMB_SIZE = 256
NHEAD = 1
WORD_LEN = 16

#train
EPOCH = 100
BATCH_SIZE = 5
DEVCIE = 'cuda' if torch.cuda.is_available() else 'cpu'