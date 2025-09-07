import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class Graph_Generator(nn.Module):
    def __init__(self,model_dim, spatial_embedding=None):
        super().__init__()

        S=60
        self.node_U1 = nn.init.xavier_normal_(
            nn.Parameter(
                torch.FloatTensor(model_dim, S)
            )
        )

        self.node_U2 = nn.init.xavier_normal_(
            nn.Parameter(
                torch.FloatTensor(S, model_dim)
            )
        )
        self.spatial = spatial_embedding
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        Exa_attn = torch.matmul(x, self.node_U1)

        Exa_attn = self.softmax(Exa_attn)

        Exa_x = torch.matmul(Exa_attn, self.node_U2)
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("btnc, mc->bnm", x, self.spatial).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bnc, bmc->bnm", Exa_x.sum(1), Exa_attn.sum(1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        # adj_dyn = (adj_dyn_1 + adj_dyn_2 + adj)/2
        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = mask

        return adj_f

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)




        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x,mask):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)


        query, key, value=x,x,x
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]




        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)


        mask=mask.unsqueeze(1).repeat(1,self.num_heads,1,1).reshape(-1,tgt_length,tgt_length)

        print(mask)



        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)


        attn_score=attn_score

        if self.mask:
             attn_score=attn_score*mask

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

"""
x=torch.randn(64,12,170,64)

mask=Graph_Generator(64,170)

attn=AttentionLayer(64,4)

mask=mask(x)

y=attn(x,mask)
print(y)
"""