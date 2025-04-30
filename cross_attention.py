import torch
import torch.nn as nn
from multiheadattention import *
from config import *

class CrossAttention(nn.Module):
    def __init__(self, model_dim, nhead, feedforward_dim):
        super(CrossAttention, self).__init__()
        self.attention = MultiHeadAttention(model_dim, nhead)
        self.norm1 = nn.LayerNorm(model_dim)

        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        self.norm2 = nn.LayerNorm(model_dim)
    
    def forward(self, query, key):
        attn_output = self.attention(query, key, key)
        attn_output = query + attn_output
        attn_output = self.norm1(attn_output)
        
        feedforward_output = self.linear1(attn_output)
        feedforward_output = self.activation(feedforward_output)
        feedforward_output = self.linear2(feedforward_output)

        output = attn_output + feedforward_output
        output = self.norm2(output)

        return output

if __name__ == "__main__":
    model_dim = 256
    nhead = NHEAD
    feedforward_dim = model_dim * 2
    cross_attention = CrossAttention(model_dim, nhead, feedforward_dim)

    batch_size = 10
    query_len = 20
    key_len = 30
    query = torch.randn(batch_size, query_len, model_dim)
    key = value = torch.randn(batch_size, key_len, model_dim)
    
    output = cross_attention(query, key)
    print(output.shape)