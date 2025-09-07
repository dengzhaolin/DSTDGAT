import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import time


class LearnableGlobalLocalMultiheadAttention(nn.Module):
    NUM_WEIGHTS = 7

    def __init__(
            self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_k = self.bias_v = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    # global
    def in_proj_global_q(self, query):
        return self._in_proj(query, start=0, end=self.embed_dim)

    def in_proj_global_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_global_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim, end=3 * self.embed_dim)

    # local mask
    def in_proj_local_mask_q(self, query):
        return self._in_proj(query, start=3 * self.embed_dim, end=4 * self.embed_dim)

    def in_proj_local_mask_k(self, key):
        return self._in_proj(key, start=4 * self.embed_dim, end=5 * self.embed_dim)

    # local right
    def in_proj_local_q(self, query):
        return self._in_proj(query, start=5 * self.embed_dim, end=6 * self.embed_dim)

    def in_proj_local_k(self, key):
        return self._in_proj(key, start=6 * self.embed_dim, end=7 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def scan_local_masking(self, q_mask, k_mask):


        attention = torch.bmm(q_mask, k_mask.transpose(1, 2))
        time_len=attention.shape[1]
        key = attention.clone()
        scr_size = attention.size()
        _, max_indices = torch.max(attention, dim=-1)
        key = F.relu(key)
        max_indices = max_indices.unsqueeze(2).expand(*max_indices.shape,attention.shape[-1])



        dis_mask = torch.ones(time_len, device=q_mask.device, dtype=q_mask.dtype)

        for i in range(time_len):
            dis_mask[i] = i
        dis_mask = dis_mask.expand(attention.shape[0],time_len,time_len)




        dis_mask = F.normalize(torch.abs(dis_mask - max_indices), p=1, dim=-1)
        mean_value = torch.mean((key * dis_mask), dim=-1) * scr_size[-1] / (scr_size[-1] - 1)

        mean_value = mean_value.unsqueeze(2).expand(*attention.shape)
        key = dis_mask * attention
        query = key - mean_value


        query = torch.where(query < 0, torch.tensor(0, device=q_mask.device, dtype=q_mask.dtype),
                            torch.tensor(1, device=q_mask.device, dtype=q_mask.dtype))

        """


        for idx, x in enumerate(query):
            for idy, y in enumerate(x):
                index = max_indices[idx][idy][0]
                query[idx, idy, index] = 1
                
        """


        return query

    def forward(self, query, key,  value):

        query=query.permute(1,0,2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()


        q = self.in_proj_global_q(query)
        k = self.in_proj_global_k(key)
        v = self.in_proj_global_v(value)
        q_mask = self.in_proj_local_mask_q(query)
        k_mask = self.in_proj_local_mask_k(key)
        q_local = self.in_proj_local_q(query)
        k_local = self.in_proj_local_k(key)

        q = q * self.scaling
        q_local = q_local * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_local = q_local.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)


        k_local = k_local.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_mask = k_mask.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_mask = q_mask.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)


        global_attn_weights = torch.bmm(q, k.transpose(1, 2))


        scan_mask  = torch.ones(tgt_len, tgt_len, device=q.device, dtype=torch.bool).tril_()


        global_attn_weights = global_attn_weights.masked_fill_(~scan_mask, -torch.inf)

        local_attn_weights = torch.bmm(q_local, k_local.transpose(1, 2))


        local_att_mask = self.scan_local_masking(q_mask, k_mask)


        masked_local_attn_weights = local_attn_weights * local_att_mask
        masked_local_attn_weights = masked_local_attn_weights .masked_fill_(~scan_mask, -torch.inf)


        attn_weights = global_attn_weights# + masked_local_attn_weights  # 0.1
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)


        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn).permute(1,0,2)
        consistent_mask = torch.sum(local_att_mask, dim=0)

        return attn, consistent_mask


class Temporal_Aware_Learning(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0.
    ):
        super().__init__()

        self.attn = LearnableGlobalLocalMultiheadAttention(model_dim, num_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):

        b,l,n,c=x.shape

        residual = x

        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)

        x=x.reshape(b*n,l,c)
        out,_ = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)

        out=out.permute(1,0,2)
        out = self.dropout1(out)
        out=out.reshape(b,l,n,c)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        return out

"""
ATTN=Temporal_Aware_Learning(64)
x=torch.randn(64,12,325,64)
y=ATTN(x,1)
print(y)

"""




