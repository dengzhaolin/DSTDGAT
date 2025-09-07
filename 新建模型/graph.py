import torch
from torch import nn
import torch.nn.functional as F
import math
from .PGAT import *

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim,node_num,dropout=0.1):
        super().__init__()
        self.complex_weight_adaptive = nn.Parameter(torch.randn(node_num, node_num),requires_grad=True)
        self.complex_weight = nn.Parameter(torch.randn(node_num, node_num),requires_grad=True)

        self.dim = dim #// 4

        self.W=nn.Parameter(torch.randn(dim, self.dim))
        nn.init.xavier_uniform_(self.W)

        nn.init.trunc_normal_(self.complex_weight_adaptive)
        nn.init.trunc_normal_(self.complex_weight)

        self.memory=nn.Parameter(torch.randn(node_num, dim))

        nn.init.trunc_normal_(self.memory)
        self.a = nn.Parameter(torch.rand(2 * self.dim, 1))



        self.dropout = nn.Dropout(dropout)

        self.fc=nn.Linear(2,1)

        self.adaptive_graph=True

        self.leakyrelu=nn.LeakyReLU()



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

        #h = torch.matmul(x_in, self.W).sum(1)
        h = x_in.sum(1)




        #a_input = torch.cat([h.repeat(1, 1, N).view(B,  N * N, -1), h.repeat(1, N, 1)], dim=-1).view(B, N, -1, 2 * self.dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        Wh1 = torch.matmul(h,self.a[:self.dim, :])
        Wh2 = torch.matmul(h,self.a[self.dim:, :])  # N*1

        e = Wh1 + Wh2.permute(0,2,1)
        # [B, N, N, 2 * out_features]

        #attention = self.dropout(F.softmax(e, dim=-1))  # [N, N]！
        attention =e# [N, N]！



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
class Adaptive_Spectral_Block1(nn.Module):
    def __init__(self, dim,node_num,dropout=0.1):
        super().__init__()
        self.complex_weight_adaptive = nn.Parameter(torch.randn(node_num, node_num),requires_grad=True)
        self.complex_weight = nn.Parameter(torch.randn(node_num, node_num),requires_grad=True)

        self.dim = dim #// 4

        self.W=nn.Parameter(torch.randn(dim, self.dim))
        nn.init.xavier_uniform_(self.W)

        nn.init.trunc_normal_(self.complex_weight_adaptive)
        nn.init.trunc_normal_(self.complex_weight)

        self.memory=nn.Parameter(torch.randn(node_num, dim))

        nn.init.trunc_normal_(self.memory)
        self.a = nn.Parameter(torch.rand(2 * self.dim, 1))



        self.dropout = nn.Dropout(dropout)

        self.fc=nn.Linear(2,1)

        self.adaptive_graph=True

        self.leakyrelu=nn.LeakyReLU()



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

        adj_f=self.complex_weight_adaptive*adj_dyn_1 +self.complex_weight*adj_dyn_2

        adj_f=F.softmax(F.relu(adj_f),dim=-1)


        return adj_f

    def forward(self, x_in):

        x_in=x_in.permute(0,3,2,1)

        B, L, N, C = x_in.shape

        h = torch.matmul(x_in, self.W)
        h_ = h.sum(1)


        #a_input = torch.cat([h.repeat(1, 1, N).view(B,  N * N, -1), h.repeat(1, N, 1)], dim=-1).view(B, N, -1, 2 * self.dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        Wh1 = torch.matmul(h_,self.a[:self.dim, :])
        Wh2 = torch.matmul(h_,self.a[self.dim:, :])  # N*1

        e = Wh1 + Wh2.permute(0,2,1)
        # [B, N, N, 2 * out_features]

        #attention = self.dropout(F.softmax(e, dim=-1))  # [N, N]！
        attention =self.leakyrelu(e)# [N, N]！


        if self.adaptive_graph:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)

            #h_ada = torch.matmul(x_in, self.W)

            adj_f=self.create_adaptive__mask(h)

            topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
            mask = torch.zeros_like(adj_f)
            mask.scatter_(-1, topk_indices, 1)

            zero_vec = -1e12 * torch.ones_like(e)
            attention = torch.where(mask > 0, attention, zero_vec)  # [B,N, N]


        attention=attention

        attention=self.dropout(F.softmax(attention,dim=-1))


        h_prime = torch.einsum("btnc,bnm->bcmt", h, attention).contiguous()
        h_prime=F.relu(h_prime)


        return h_prime#
class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output
class Diffusion_GAT(nn.Module):
    def __init__(self, channels=128, diffusion_head=1,num_nodes=170, dropout=0.1):
        super().__init__()
        self.diffusion_head = diffusion_head
        self.conv = nn.Conv2d(diffusion_head * channels, channels, (1, 1))
        #self.generator = Adaptive_Spectral_Block(channels, num_nodes, dropout)  # GEANet(channels,4)

        self.generator = nn.ModuleList([Adaptive_Spectral_Block(channels, num_nodes, dropout)
                                        for _ in range(diffusion_head)])
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        out = []

        for i in range(0, self.diffusion_head):
            adj=self.generator[i](x)
            if adj.dim() == 3:
                x_head = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x_head)
            elif adj.dim() == 2:
                x_head = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x_head)


        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output
class Diffusion_GAT1(nn.Module):
    def __init__(self, channels=128, diffusion_head=1,num_nodes=170, dropout=0.1):
        super().__init__()
        self.diffusion_head = diffusion_head
        self.conv = nn.Conv2d(diffusion_head * channels, channels, (1, 1))
        #self.generator = Adaptive_Spectral_Block(channels, num_nodes, dropout)  # GEANet(channels,4)

        self.generator = nn.ModuleList([Adaptive_Spectral_Block1(channels, num_nodes, dropout)
                                        for _ in range(diffusion_head)])
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        out = []


        for i in range(0, self.diffusion_head):
            x_head=self.generator[i](x)

            out.append(x_head)

        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output
class DGCN1(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=2, dropout=0.1, emb=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator =Adaptive_Spectral_Block1(channels,num_nodes,dropout)# GEANet(channels,4)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x):
        skip = x
        x = self.conv(x)

        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x* self.emb  + skip#
        return x
class DGCN2(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_head=4, dropout=0.1, emb=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.gat = Diffusion_GAT(channels, diffusion_head,num_nodes, dropout)




        self.emb = emb

    def forward(self, x):
        skip = x
        x = self.conv(x)
        x = self.gat(x)
        x = x* self.emb  + skip#
        return x
"""
emb=torch.randn(64,170,12)
dgcn=DGCN(64,170,1,0.1,emb)

x=torch.randn(64,64,170,12)

y=dgcn(x)
print(y.shape)
"""