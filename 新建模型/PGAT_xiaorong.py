

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim,node_num,dropout=0.1):
        super().__init__()
        self.complex_weight_adaptive = nn.Parameter(torch.randn(node_num, node_num),requires_grad=True)
        self.complex_weight = nn.Parameter(torch.randn(node_num, node_num),requires_grad=True)

        self.dim = dim #// 4

        nn.init.trunc_normal_(self.complex_weight_adaptive)
        nn.init.trunc_normal_(self.complex_weight)

        self.memory=nn.Parameter(torch.randn(node_num, dim))

        nn.init.trunc_normal_(self.memory)
        self.a = nn.Parameter(torch.rand(2 * self.dim, 1))



        self.dropout = nn.Dropout(dropout)

        self.fc=nn.Linear(2,1)

        self.adaptive_graph=True

        self.leakyrelu=nn.LeakyReLU(0.1)



        #self.a = nn.Parameter(torch.FloatTensor(size=(2 * self.dim, 1)))

    def create_adaptive__mask(self, x_in):
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("btnc, mc->bnm", x_in, self.memory).contiguous()
                / math.sqrt(x_in.shape[-1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bnc, bmc->bnm", x_in.sum(1), x_in.sum(1)).contiguous()
                / math.sqrt(x_in.shape[-1])
            ),
            -1,
        )

        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        #adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)
        adj_f = self.fc(adj_f).squeeze()


        return adj_f

    def forward(self, x_in):

        x_in=x_in.permute(0,3,2,1)

        B, L, N, C = x_in.shape

        h = x_in.sum(1)

        Wh1 = torch.matmul(h,self.a[:self.dim, :])
        Wh2 = torch.matmul(h,self.a[self.dim:, :])  # N*1

        e = Wh1 + Wh2.permute(0,2,1)
        # [B, N, N, 2 * out_features]

        #attention = self.dropout(F.softmax(e, dim=-1))  # [N, N]！
        attention =e



        weight = self.complex_weight

        adj_weighted = attention * weight



        if self.adaptive_graph:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)

            #h_ada = torch.matmul(x_in, self.W)

            adj_f=self.create_adaptive__mask(x_in)

            weight_adaptive = self.complex_weight_adaptive
            adj_weighted2 = adj_f * weight_adaptive

            adj_weighted += adj_weighted2

        adj_weighted=self.dropout(F.softmax(adj_weighted,dim=-1))



        topk_values, topk_indices = torch.topk(adj_weighted, k=int(N* 0.8), dim=-1)
        mask = torch.zeros_like(adj_weighted)
        mask.scatter_(-1, topk_indices, 1)

        adj_weighted = adj_weighted * mask
        


        return adj_weighted#
class Diffusion_GAT2(nn.Module):
    def __init__(self, channels=128, diffusion_head=1,num_nodes=170, dropout=0.1,emb=None,cheb_polynomials=None):
        super().__init__()
        self.diffusion_head = diffusion_head
        self.emb=emb

        self.W = nn.Conv2d(channels, channels, (1, 1))
        self.conv = nn.Conv2d(diffusion_head *channels, channels, (1, 1))#
        #self.generator = Adaptive_Spectral_Block(channels, num_nodes, dropout)  # GEANet(channels,4)

        self.generator = nn.ModuleList([Adaptive_Spectral_Block(channels, num_nodes, dropout)
                                        for _ in range(diffusion_head)])
        self.dropout = nn.Dropout(dropout)

        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(channels,channels)) for _ in range(diffusion_head)])

        self.cheb_polynomials = cheb_polynomials



    def forward(self, x):
        out = []

        skip = x

        x = self.W(x)



        for i in range(0, self.diffusion_head):
            T_k = torch.tensor(self.cheb_polynomials[i],dtype=torch.float32).to(x.device)

            theta_k = self.Theta[i]

            #adj=T_k.mul(self.generator[i](x))
            adj = self.generator[i](x)
            if adj.dim() == 3:
                x_head = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
            elif adj.dim() == 2:
                x_head = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()

            x_head=torch.einsum("bcnt,cd->bdnt", x_head, theta_k).contiguous()
            out.append(x_head)

        x = torch.cat(out, dim=1)

        out = self.conv(x)
        output = self.dropout(out)
        output = output* self.emb  + skip  #
        return output



"""
adj=np.array(torch.randn(170,170))
adj=cheb_polynomial(adj,4)
emb=nn.Parameter(torch.randn(64, 170 ,12))
DGCN2=Diffusion_GAT(64,4,170,0.1,emb,adj)

x=torch.randn(64,64,170,12)

y=DGCN2(x)
#print(y)
"""

