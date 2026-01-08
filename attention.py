import torch
from torch import nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self , d_model , num__head , dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num__head
        assert d_model % num__head == 0 , "d_model is not divisible by num_head"

        self.d_k = d_model // num__head
        self.w_q = nn.Linear(d_model , d_model) # w_q
        self.w_v = nn.Linear(d_model , d_model) # w_v
        self.w_k = nn.Linear(d_model , d_model) # w_k

        self.w_o = nn.Linear(d_model , d_model) # w_o

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query , key , value , mask , dropout):
        d_k = query.shape[-1]

        attn_score = (query @ key.transpose(-2 , -1)) / math.sqrt(d_k)

        if mask is not None:
            attn_score.masked_fill(mask == 0 , -1e9)
        
        attn_score = attn_score.softmax(dim = -1)

        if dropout is not None:
            attn_score = dropout(attn_score)

        return (attn_score @ value) , attn_score


    def forward(self , k , q , v , mask):
        query = self.w_q(q) # (batch , seq_len , d_model) -- > (batch , seq_len , d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch , seq_len , d_model) --> (batch , seq_len , num_head , d_k) --> (batch , num_head , seq_len , d_k)
        query = query.view(query.shape[0] , query.shape[1] , self.num_head , self.d_k).transpose(1 , 2)
        key = key.view(key.shape[0] , key.shape[1] , self.num_head , self.d_k).transpose(1 , 2)
        value = value.view(value.shape[0] , value.shape[1] , self.num_head , self.d_k).transpose(1 , 2)

        x , attn_score = MultiHeadAttention.attention(query , key , value , mask , self.dropout)

        # (batch , num_head , seq_len , d_k) --> (batch , seq_len , num_head , d_k) --> (batch , seq_len , d_model)
        x = x.transpose(1 , 2).contiguous().view(x.shape[0] , -1 , self.num_head * self.d_k)

        # (batch , seq_len , d_model) --> (batch , seq_len , d_model)
        return self.w_o(x)