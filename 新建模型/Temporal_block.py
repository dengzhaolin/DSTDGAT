import torch
import torch.nn as nn
from torch import Tensor
from torch import nn, sum
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange
from torch.nn.utils import weight_norm
from mamba_ssm import Mamba
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

    def forward(self, x):
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



        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)


        attn_score=attn_score

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out



class Temporal_Attention(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0., mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(2*model_dim, num_heads, mask)

        self.gru_forward = nn.GRU(model_dim, model_dim, num_layers=2, batch_first=True)
        self.gru_backward = nn.GRU(model_dim, model_dim, num_layers=2, batch_first=True)

        self.norm_forward = nn.LayerNorm(model_dim)
        self.norm_backward = nn.LayerNorm(model_dim)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x1):
        b, l, n, c = x1.shape

        x1 = x1.transpose(1, -2)

        x1 = x1.reshape(b * n, l, c)

        # x: (batch_size, ..., length, model_dim)
        residual = x1

        out_forward,(_,_) = self.gru_forward(x1)

        out_forward=self.norm_forward(out_forward)

        out_backward,(_,_) = self.gru_backward(torch.flip(x1,dims=[1]))

        out_backward=self.norm_backward(out_backward)



        bid_out = torch.cat((out_forward, out_backward), dim=-1)

        out = self.attn(bid_out)

        z = self.sigmoid(out)

        out = self.tanh1(out_forward) * z + self.tanh2(out_backward) * (1 - z)

        out = self.dropout1(out)

        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.reshape(b, n, l, c)

        out = out.transpose(1, -2)
        return out





class TemporalExternalAttn(nn.Module):
    def __init__(self, d_model, S=256):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model
        return out
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))


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


class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
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
        adj_f = adj_f * mask

        return adj_f
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

class DGCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator = GEANet(channels,4)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x,spatial_embedding):
        skip = x
        x = self.conv(x)

        adj_dyn = self.generator(x.permute(0,3,2,1),spatial_embedding)
        x = self.gcn(x, adj_dyn)
        x = x* self.emb  + skip #
        return x


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1,1))
        self.conv2 = nn.Conv2d(features, features, (1,1))
        self.conv3 = nn.Conv2d(features, features, (1,1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class TemporLearning(nn.Module):
    def __init__(self,model_dim,feed_forward_dim,num_node,num_head,dropout,emb1,emb2):
        super().__init__()

        channels=model_dim


        self.even_DS = GLU(channels, dropout)

        self.even_Attn=AttentionLayer(channels,num_head)


        self.odd_DS = GLU(channels, dropout)

        self.odd_Attn = AttentionLayer(channels, num_head)
        self.conv = nn.Conv1d(channels, channels, 1)

        self.gru=nn.GRU(channels,channels*2,num_layers=2,batch_first=True)



        self.feed_forward1 = nn.Sequential(
            nn.Linear(model_dim*2, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        self.gcn1=DGCN(channels,1,0.1,emb1)
        self.gcn2= DGCN(channels,1,0.1,emb2)

        #self.gcn1=SelfAttentionLayer1(channels,feed_forward_dim,num_head,dropout,emb1)
        #self.gcn2=SelfAttentionLayer1(channels,feed_forward_dim,num_head,dropout,emb2)

        #self.gcn1 = AttentionLayer_Spa(model_dim, num_node, 20, num_head, mask=None, embed=emb1)
        #self.gcn2 = AttentionLayer_Spa(model_dim, num_node, 20, num_head, mask=None, embed=emb2)




    def forward(self, x,spatial_embedding):

        x=x.permute(0, 2, 3, 1)

        b, l, n, c = x.shape

        x1 = x.permute(0, 2, 3, 1)  ## B N C L

        #x2 = x.permute(0, 3, 2, 1) ##
        x2=x  ##B L N C

        x1 = x1.reshape(b * n, c, l)

        x_gru = self.conv(x1).permute(0, 2, 1)  ##b*n l c

        out_gru,(_,_) = self.gru(x_gru)

        out_gru=out_gru.reshape(b,l,n,-1)


        x_even, x_odd = x2 ,torch.flip(x2,dims=[1])

        x_even_attn = self.even_Attn(x_even)


        #x_even_attn=self.gcn1(x_even_attn,spatial_embedding)
        #x_even_attn = self.gcn1(x_even_attn)


        x_even_ds = self.even_DS(x_even.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)


        x_even = (x_even_attn * x_even_ds)

        x_odd_attn = self.odd_Attn(x_odd)


       # x_odd_attn=self.gcn2(x_odd_attn,spatial_embedding)
        #x_odd_attn = self.gcn2(x_odd_attn)

        x_odd_ds = self.odd_DS(x_odd.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        x_odd = (x_odd_attn * x_odd_ds)

        out_att_ds=torch.cat((x_even,x_odd),dim=-1) ## b l n c


        out=out_att_ds+out_gru

        out = self.feed_forward1(out)

        out = self.dropout1(out)

        out = self.ln2(x+ out)

        residual=out

        out=self.gcn1(out.permute(0, 3, 2, 1),spatial_embedding).permute(0, 3, 2, 1)

        out = self.feed_forward2(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        x = x.permute(0, 2, 3, 1)

        return out

class Sem_Graph_Encoder1(nn.Module):
    def __init__(self,model_dim,num_layer,num_node,adj,culter_size,edge, feed_forward_dim=2048, num_heads=8, dropout=0, mask=None,dtw_mask=None
    ):
        super().__init__()

        emb1 = nn.Parameter(torch.randn(model_dim, num_node, 12))
       # emb1 = nn.Parameter(torch.randn(num_node,model_dim//num_heads))
        #emb2 = nn.Parameter(torch.randn(num_node, model_dim//num_heads))
        emb2=nn.Parameter(torch.randn(model_dim, num_node, 12))
        self.sem_attention = nn.ModuleList(
            [
                TemporLearning(model_dim,  feed_forward_dim,num_node, num_heads, dropout,emb1,emb2
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



class AttentionLayer_Spa(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim,num_node,cluter_size, num_heads=8, mask=None,embed=None):
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

        self.embed= embed

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

        self.linear=nn.Linear(self.head_dim,self.head_dim)

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

        spik=extenal_value

        Exa_attn = torch.matmul(extenal_value,self.node_U1)

        Exa_attn = self.softmax(Exa_attn)

        Exa_Value=torch.matmul(Exa_attn,self.node_U2)

        Exa_Value = Exa_Value * self.embed  + spik








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
class SelfAttentionLayer1(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0., embed=None
    ):
        super().__init__()

        self.attn = AttentionLayer_Spa(model_dim,170,20, num_heads,mask=None,embed=embed)
        #self.attn = Attention_Temporal(model_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x1):



        residual = x1

        out = self.attn(x1)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        return out





"""
x=torch.rand(64,12,325,64)

at=TemporLearning(64,4,0.1)

y=at(x)

print(y.shape)

"""


