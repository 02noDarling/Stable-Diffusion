import torch

# model
IMG_SIZE = 48
T = 1000
EMB_SIZE = 256
NHEAD = 1
WORD_LEN = 16

# train
EPOCH = 2
BATCH_SIZE = 100
DEVCIE = 'cuda' if torch.cuda.is_available() else 'cpu'

# lora
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 1
LORA_FINETUNE_LAYERS = ['q_linear', 'k_linear', 'v_linear', 'out_linear', 'linear1', 'linear2']