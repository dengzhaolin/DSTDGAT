import logging
import math
import torch
import torch.nn as nn
class GraphPropagationAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = node_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(node_dim, node_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(node_dim, node_dim)

        self.reduce = nn.Conv2d(edge_dim, num_heads, kernel_size=1)
        self.expand = nn.Conv2d(num_heads, edge_dim, kernel_size=1)
        if edge_dim != node_dim:
            self.fc = nn.Linear(edge_dim, node_dim)
        else:
            self.fc = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, node_embeds, edge_embeds, padding_mask):
        # node-to-node propagation
        B, N, C = node_embeds.shape
        qkv = self.qkv(node_embeds).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, n_head, 1+N, 1+N]
        attn_bias = self.reduce(edge_embeds)  # [B, C, 1+N, 1+N] -> [B, n_head, 1+N, 1+N]
        attn = attn + attn_bias  # [B, n_head, 1+N, 1+N]
        residual = attn

        #attn = attn.masked_fill(padding_mask, float("-inf"))
        attn = attn.softmax(dim=-1)  # [B, C, N, N]
        attn = self.attn_drop(attn)
        node_embeds = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # node-to-edge propagation
        edge_embeds = self.expand(attn + residual)  # [B, n_head, 1+N, 1+N] -> [B, C, 1+N, 1+N]

        # edge-to-node propagation
       # w = edge_embeds.masked_fill(padding_mask, float("-inf"))
        w=edge_embeds
        w = w.softmax(dim=-1)
        w = (w * edge_embeds).sum(-1).transpose(-1, -2)
        node_embeds = node_embeds + self.fc(w)
        node_embeds = self.proj(node_embeds)
        node_embeds = self.proj_drop(node_embeds)

        return node_embeds, edge_embeds


g=GraphPropagationAttention(64,64)

x=torch.rand(64,325,64)
edge=torch.rand(64,64,325,325)

node_embeds, edge_embeds=g(x,edge,None)
print(node_embeds.shape)
print(edge_embeds.shape)