import torch
from torch import Tensor
from torch import nn, sum
from torch.nn import init
import torch.nn.functional as F
import math
class AttentionLayer_Spa(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim,num_node,edge,cluter_size, num_heads=8, mask=None):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)


        self.out_proj = nn.Linear(model_dim, model_dim)

        self.adp_pos =nn.Parameter(torch.zeros(num_node,cluter_size))
        self.cluter_k = nn.AdaptiveAvgPool2d((cluter_size, None))
        self.cluter_v = nn.AdaptiveAvgPool2d((cluter_size, None))

    def forward(self,x):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        query, key, value=x,x,x
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        key = self.cluter_k(key)  ### b l m c   m = cluter_dim

        value = self.FC_V(value)
        value = self.cluter_v(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)


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








        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        #attn_out = out

        #out = out + f


        out = self.out_proj(out)

        return out#,attn_out

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
        attn = attn / sum(attn, dim=self.dim2, keepdim=True)  # bs,n,S
        return attn
class GEANet(nn.Module):

    def __init__(
            self, dim, n_heads):
        super().__init__()

        self.dim = dim
        self.external_num_heads =n_heads
        self.unit_size = self.dim//n_heads

        # self.q_Linear = nn.Linear(in_dim, gconv_dim - dim_pe)
        self.node_U1 = nn.Linear(self.unit_size, self.unit_size)
        self.node_U2 = nn.Linear(self.unit_size, self.unit_size)

        assert self.unit_size * self.external_num_heads == self.dim, "dim must be divisible by external_num_heads"

        self.fc = nn.Linear(2, 1)


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
        node_out = self.node_U1(node_out)
        attn = self.norm(node_out)  # 行列归一化  N x 16 x 4
        node_out = self.node_U2(attn)
        node_out = node_out.reshape(b, l, N, -1)

        node_x=node_x.permute(0, 3, 2, 1)

        node_out=node_out.permute(0, 3, 2, 1)

        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, mc->bnm", node_x, embedding).contiguous()
                / math.sqrt(node_x.shape[1])
            ),
            -1,
        )

        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", node_out.sum(-1), node_out.sum(-1)).contiguous()
                / math.sqrt(node_out.shape[1])
            ),
            -1,
        )

        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)

        adj_f = adj_f * mask

        return adj_f



class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x=x.permute(0,3,2,1)
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
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


        adj_f=adj_f*mask


        return adj_f


class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels//2, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x=x.permute(0,3,2,1)
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
        output = output.permute(0, 3, 2, 1)
        return output

class SelfAttentionLayer1(nn.Module):
    def __init__(
        self, model_dim,num_node,edge,culter_size, feed_forward_dim=2048, num_heads=8, dropout=0, mask=None,dtw_mask=None
    ):
        super().__init__()


        self.attn_S = AttentionLayer_Spa(model_dim,num_node,edge,culter_size, num_heads, mask)

        self.extra_attn = GEANet(model_dim, num_heads, True, model_dim, model_dim)

        self.graph=Graph_Generator(model_dim*2,num_node)

        self.gcn=Diffusion_GCN(model_dim*2,2)

        self.feed_forward1 = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )




        self.ln1 = nn.LayerNorm(model_dim)

        self.ln2 = nn.LayerNorm(model_dim)



        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        self.spatial_linear=nn.Linear(model_dim,model_dim)

        self.spatial_norm=nn.LayerNorm(model_dim)




    def forward(self, x,spatial_embed):
        # x: (batch_size, length,node, model_dim)

        extra_attn_spa = self.extra_attn(x, spatial_embed)

        residual = x

        out_attn_s= self.attn_S(x)  # (batch_size, ..., length, model_dim)

        attn=torch.cat((extra_attn_spa,out_attn_s),dim=-1)

        graph=self.graph(attn)

        out_attn=self.gcn(attn,graph)

        out_attn=self.dropout1(out_attn)

        out = self.ln1(residual + out_attn)

        residual = out

        out_attn = self.feed_forward1(out) # (batch_size, ..., length, model_dim)

        out_attn = self.dropout2(out_attn)

        out_attn = self.ln2(out_attn+residual)

        return out_attn


class Sem_Graph_Encoder1(nn.Module):
    def __init__(self,model_dim,num_layer,num_node,adj,culter_size,edge, feed_forward_dim=2048, num_heads=8, dropout=0, mask=None,dtw_mask=None
    ):
        super().__init__()

        self.sem_attention = nn.ModuleList(
            [
                SelfAttentionLayer1(model_dim, num_node, edge,culter_size, feed_forward_dim, num_heads, dropout, mask,dtw_mask
                                   )
                for _ in range(num_layer)
            ]
        )

        self.adj=adj


        self.layer=num_layer

        self.edge_embedding=nn.Linear(num_node,model_dim)





    def forward(self,x):

        spatial_embedding = self.edge_embedding(self.adj)


        for i in range(self.layer):

            x =self.sem_attention[i](x,spatial_embedding)


        return x