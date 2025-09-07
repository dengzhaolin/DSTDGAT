
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, num_nodes, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_nodes = num_nodes

        self.W = nn.Parameter(torch.randn(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W)
        self.meta_memory=nn.Parameter(torch.randn(size=(out_features, num_nodes)))


        self.downsample = nn.Linear(in_features, out_features)

        self.bias = nn.Parameter(torch.zeros(num_nodes, out_features))

    def forward(self, input):

        input=input.permute(0,3,2,1)

        h = torch.matmul(input, self.W)

        adp_e=torch.matmul(self.meta_memory.T,self.meta_memory)

        e = F.softmax(F.relu(adp_e), dim=-1)

        attention=e


        h_prime = torch.einsum("btnc,nm->btmc", h,attention).contiguous()

        if input.shape[-1] != h_prime.shape[-1]:
            input_transformed = self.downsample(input)
            h_prime = h_prime + input_transformed
        else:
            h_prime = h_prime + input


        return h_prime.permute(0,3,2,1)

class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128,num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.gat = GraphAttentionLayer(channels, channels, num_nodes, dropout, 0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []

        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x_gcn = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                x_gat=self.gat(x)
                out.append(x_gcn+x_gat)
            elif adj.dim() == 2:
                x_gcn = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                x_gat = self.gat(x)

                out.append(x_gcn + x_gat)

        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output


class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )

        adj_f = torch.softmax(adj_dyn_2, -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = adj_f * mask

        return adj_f

class DGCN3(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator =Graph_Generator(channels,num_nodes,diffusion_step,dropout)# GEANet(channels,4)
        self.gcn = Diffusion_GCN(channels,num_nodes, diffusion_step, dropout)

        self.emb = emb

    def forward(self, x):
        skip = x
        x = self.conv(x)

        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x * self.emb + skip#
        return x
"""
emb=nn.Parameter(torch.randn(64, 170 ,12))

DGCN3=DGCN3(64,170,2,0.1,emb)

x=torch.randn(64,64,170,12)

y=DGCN3(x)
print(y)
"""
