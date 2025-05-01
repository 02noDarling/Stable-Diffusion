import torch

# model
IMG_SIZE = 48
T = 1000
EMB_SIZE = 256
NHEAD = 1
WORD_LEN = 16

#train
EPOCH = 1
BATCH_SIZE = 1
DEVCIE = 'cuda' if torch.cuda.is_available() else 'cpu'