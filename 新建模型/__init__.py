
import torch
import torch.nn as nn
from torch import Tensor
from torch import nn, sum
from torch.nn import init
import torch.nn.functional as F
import math
class DNorm(nn.Module):
    def __init__(
            self,
            dim1=-2, dim2=-1
    ):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.softmax = nn.Softmax(dim=self.dim1)  # N

    def forward(self, attn: Tensor) -> Tensor:
        # softmax = nn.Softmax(dim=0)  # N
        attn = self.softmax(attn)  # bs,n,S
        return attn

class GEANet(nn.Module):

    def __init__(
            self, dim, n_heads):
        super().__init__()

        self.dim = dim
        self.external_num_heads =n_heads
        self.unit_size = self.dim//n_heads

        S=60

        # self.q_Linear = nn.Linear(in_dim, gconv_dim - dim_pe)
        self.node_U1 = nn.Parameter(torch.randn(self.unit_size,S))
        self.node_U2 = nn.Parameter(torch.randn(S,self.unit_size))
        assert self.unit_size * self.external_num_heads == self.dim, "dim must be divisible by external_num_heads"

        self.fc = nn.Linear(1, 1)

        meta_axis=True

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(dim,self.unit_size, dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(dim, dim))
            )


        self.norm = DNorm()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, node_x, embedding=None) -> Tensor:
        b,l,N,d=node_x.shape
        node_out = node_x.reshape(b, l, N, self.external_num_heads, -1)  # Q * 4（head）  ：  N x 16 x 4(head)
        node_out = node_out.transpose(2, 3)  # (N, 16, 4) -> (N, 4, 16)
        node_out = torch.matmul(node_out,self.node_U1)
        attn = self.norm(node_out)  # 行列归一化  N x 16 x 4
        node_out = torch.matmul(attn,self.node_U2)
        node_out = node_out.reshape(b, l, N, -1)

        if self.meta_axis:

            weights = torch.einsum(
                "nd,dio->nio", embedding, self.weights_pool
            )  # N,unite_size, out_dim
            bias = torch.matmul(embedding, self.bias_pool)
            node_out = (
                    torch.einsum("blni,nio->blno", node_out, weights) + bias
            )  # B, N, out_dim


        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", node_out.sum(-1), node_out.sum(-1)).contiguous()
                / math.sqrt(node_out.shape[1])
            ),
            -1,
        )

        adj_f = (adj_dyn_2).unsqueeze(-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)

        adj_f = adj_f * mask

        return adj_f


class AttentionLayer_Spa(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim,num_node,cluter_size, num_heads=8, mask=None):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads


        S=60

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.FC_V_Exa = nn.Linear(model_dim, model_dim)



        self.edge_embedding=nn.Linear(num_node,num_node)

        self.out_proj = nn.Linear(model_dim, model_dim)

        self.adp_pos =nn.Parameter(torch.zeros(num_node,cluter_size))
        self.cluter_k = nn.AdaptiveAvgPool2d((cluter_size, None))
        self.cluter_v = nn.AdaptiveAvgPool2d((cluter_size, None))

        self.node_U1 = nn.init.xavier_normal_(
            nn.Parameter(
                torch.FloatTensor(self.head_dim,S)
            )
        )

        self.node_U2 = nn.init.xavier_normal_(
            nn.Parameter(
                torch.FloatTensor(S,self.head_dim)
            )
        )

        self.softmax = nn.Softmax(dim=-2)

    def forward(self,x):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        query, key, value=x,x,x

        extenal_value= x
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        key = self.cluter_k(key)  ### b l m c   m = cluter_dim

        value = self.FC_V(value)
        value = self.cluter_v(value)

        extenal_value = self.FC_V_Exa(extenal_value)



        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        extenal_value = torch.cat(torch.split(extenal_value, self.head_dim, dim=-1), dim=0)

        Exa_attn = torch.matmul(extenal_value,self.node_U1)

        Exa_attn = self.softmax(Exa_attn)

        Exa_Value=torch.matmul(Exa_attn,self.node_U2)






        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)


        attn_score=attn_score + self.adp_pos.unsqueeze(0).unsqueeze(0) #+ edge_embedding.unsqueeze(0).unsqueeze(0)

        if self.mask is not  None:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

        out=out+Exa_Value




        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)


        out = self.out_proj(out)

        return out
"""
att=AttentionLayer_Spa(64,170,20)

x=torch.randn(64,12,170,64)

y=att(x)
print(y.shape)
"""