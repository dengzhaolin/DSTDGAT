import math

import torch
from torch import nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, seq_len,adj=None, mode='temporal',order=2,use_spatial_adj=True,is_adp=True , use_temporal_delay=True,
                 temporal_connection_len=0.9):
        self.nodes_ = """
        :param dim_int: Channel input dimension
        :param dim_out: Channel output dimension
        :param num_nodes: Number of nodes
        :param neighbour_num: Neighbor numbers. Used in temporal GCN to create edges
        :param mode: Either 'spatial' or 'temporal'
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param connections: Spatial connections for graph edges (Optional)
        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        self.relu = nn.ReLU()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_temporal_delay = use_temporal_delay
        self.use_spatial_adj = use_spatial_adj
        self.seq_len = seq_len
        self.order=order

        self.is_adp=is_adp

        self.norm=nn.LayerNorm(dim_out)
        self.support_len=0


        self.is_adp = is_adp

        if mode == 'spatial' and  self.use_spatial_adj:
            self.adj = adj
            if use_spatial_adj:
                self.support_len = 1
        elif mode == 'temporal' and  self.use_temporal_delay:
            self.adj = self._init_temporal_adj(self.seq_len,temporal_connection_len)
            if use_temporal_delay:
                self.support_len = 1
        if is_adp:
            self.support_len += 1
        c_in = (order * self.support_len ) * dim_in
        self.mlp = nn.Conv2d(c_in, dim_out, kernel_size=1)



    def _init_temporal_adj(self, time_length, decay_rate):
        """Connects each joint to itself and the same joint withing next `connection_length` frames."""
        Adj = torch.ones(time_length, time_length)
        for i in range(time_length):
            v = 0
            for r_i in range(i, time_length):
                idx_s_row = i
                idx_e_row = (i + 1)
                idx_s_col = (r_i)
                idx_e_col = (r_i + 1)
                Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (
                        decay_rate ** (v))
                v = v + 1
            v = 0
            for r_i in range(i + 1):
                idx_s_row = i
                idx_e_row = (i + 1)
                idx_s_col = (i - r_i)
                idx_e_col = (i - r_i + 1)
                Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (
                        decay_rate ** (v))
                v = v + 1
        return Adj


    def forward(self, x,support_adp):
        """
        x: tensor with shape [B, T, N, C]
        """
        b, t, n, c = x.shape
        if self.mode == 'temporal':
            x = x.transpose(1, 2)  # (B, T, J, C) -> (B, J, T, C)
            x = x.reshape(-1, c, t)

            if self.is_adp:
                nodevec1 = support_adp[0]
                nodevec2 = support_adp[1]
                adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=-1)
            if self.use_temporal_delay:
                time_delay = self.adj.to(x.device)
                support = [time_delay] + [adp]
            else:
                support = [adp]

        else:
            x = x.reshape(-1,c,n)

            if self.is_adp:
                nodevec1 = support_adp[0]
                nodevec2 = support_adp[1]
                adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=-1)

            if self.use_spatial_adj:
                adj = self.adj
                support = [adj] + [adp]
            else:
                support = [adp]


        out=[]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        for a in support:
            a = a.to(x.device)
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)

        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        out = out.reshape(b, t, n, -1)
        return out


class STGCN(nn.Module):
    def __init__(self,input_dim,seq_len,adj):
        super().__init__()
        self.SpaGcn = GCN(input_dim, input_dim, seq_len, adj, mode='spatial')
        self.TemGcn = GCN(input_dim, input_dim, seq_len, adj, mode='temporal')
        self.fusion = nn.Linear(input_dim * 2, 2)
        self.norm_t = nn.LayerNorm(input_dim)
        self.norm_s = nn.LayerNorm(input_dim)
        self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)

        self.fusion.bias.data.fill_(0.5)

    def forward(self,x,support_t,support_s):

        x_t = self.TemGcn(x, support_t)

        x_t = self.norm_t(x_t)

        x_s = self.SpaGcn(x, support_s)

        x_s = self.norm_s(x_s)

        alpha = torch.cat((x_t, x_s), dim=-1)

        alpha = self.fusion(alpha)

        alpha = alpha.softmax(dim=-1)

        x = x_t * alpha[..., 0:1] + x_s * alpha[..., 1:2]

        return x






"""
adj=torch.randn(325,325)
gcn=GCN(1,64,12,adj,'temporal',2)
nodevec1 = nn.Parameter(torch.randn(12, 10), requires_grad=True)
nodevec2 = nn.Parameter(torch.randn(10, 12), requires_grad=True)
x=torch.randn(64,12,325,1)
y=gcn(x,[nodevec1,nodevec2])
print(y.shape)
"""


