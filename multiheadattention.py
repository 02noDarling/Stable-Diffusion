import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
from config import *

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, nhead):
        super(MultiHeadAttention, self).__init__()
        self.model_dim =model_dim
        self.nhead = nhead
        self.k_dim = model_dim // nhead
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)
    
    def forward(self, query, key, value):
        Q = self.q_linear(query).reshape(query.shape[0], query.shape[1], self.nhead, self.k_dim).transpose(1, 2)
        K = self.k_linear(key).reshape(key.shape[0], key.shape[1], self.nhead, self.k_dim).transpose(1, 2)
        V = self.v_linear(key).reshape(value.shape[0], value.shape[1], self.nhead, self.k_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.k_dim)
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, V).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.model_dim)
        output = self.out_linear(output)
        return output

if __name__ == "__main__":
    model_dim = 256
    batch_size = 10
    query_len = 20
    key_len = 30
    attention = MultiHeadAttention(model_dim, NHEAD)
    query = torch.randn(batch_size, query_len, model_dim)
    key = value = torch.randn(batch_size, key_len, model_dim)
    output = attention(query, key, value)
    print(output.shape)