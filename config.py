import torch

IMG_SIZE = 48
T = 1000
DEVCIE = 'cuda' if torch.cuda.is_available() else 'cpu'