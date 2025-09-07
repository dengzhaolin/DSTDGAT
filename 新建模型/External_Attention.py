import torch
from torch import Tensor
from torch import nn, sum
from torch.nn import init
import torch.nn.functional as F
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
            self, dim, n_heads,meta_axis,embed_dim,output_dim):
        super().__init__()

        self.dim = dim
        self.external_num_heads =n_heads
        self.meta_axis =meta_axis
        self.unit_size = self.dim//n_heads

        # self.q_Linear = nn.Linear(in_dim, gconv_dim - dim_pe)
        self.node_U1 = nn.Linear(self.unit_size, self.unit_size)
        self.node_U2 = nn.Linear(self.unit_size, self.unit_size)

        assert self.unit_size * self.external_num_heads == self.dim, "dim must be divisible by external_num_heads"


        # nn.init.xavier_normal_(self.node_m1.weight, gain=1)
        # nn.init.xavier_normal_(self.node_m2.weight, gain=1)
        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim,dim, output_dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )



        # self.init_weights()
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

        if self.meta_axis:

            weights = torch.einsum(
                "nd,dio->nio", embedding, self.weights_pool
            )  # N,unite_size, out_dim
            bias = torch.matmul(embedding, self.bias_pool)
            node_out = (
                    torch.einsum("blni,nio->blno", node_out, weights) + bias
            )  # B, N, out_dim




        return node_out

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

        self.edge=edge

        self.edge_embedding=nn.Linear(num_node,num_node)

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


        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim,num_node,edge,culter_size, feed_forward_dim=2048, num_heads=8, dropout=0, mask=None,dtw_mask=None
    ):
        super().__init__()


        self.attn_S = AttentionLayer_Spa(model_dim,num_node,edge,culter_size, num_heads, mask)

        self.extra_attn_s = GEANet(model_dim, num_heads, True, model_dim, model_dim)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

        self.dtw_attn = DTW_AttentionLayer(model_dim, num_heads=num_heads, mask=dtw_mask)

        self.ln1 = nn.LayerNorm(model_dim)

        self.ln2 = nn.LayerNorm(model_dim)



        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        self.spatial_linear=nn.Linear(model_dim,model_dim)

        self.spatial_norm=nn.LayerNorm(model_dim)






    def forward(self, x,spatial_embed):
        # x: (batch_size, length,node, model_dim)

        extra_attn_spa = self.extra_attn_s(x, spatial_embed)

        dtw_attn=self.dtw_attn(x)

        residual = x

        out_attn_s = self.attn_S(x)  # (batch_size, ..., length, model_dim)

        out_attn = self.dropout1(out_attn_s + extra_attn_spa+dtw_attn)#

        out = self.ln1(residual + out_attn)

        residual = out

        out_attn = self.feed_forward1(out) # (batch_size, ..., length, model_dim)

        out_attn = self.dropout2(out_attn)

        out_attn = self.ln2(out_attn+residual)

        return out_attn
class DTW_AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=None):
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

        if self.mask is not None:
            mask = torch.where(self.mask < 0, torch.tensor(0, device=attn_score.device, dtype=attn_score.dtype),
                               torch.tensor(1, device=attn_score.device, dtype=attn_score.dtype)).bool()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class Sem_Graph_Encoder1(nn.Module):
    def __init__(self,model_dim,num_layer,num_node,adj,culter_size,edge, feed_forward_dim=2048, num_heads=8, dropout=0, mask=None,dtw_mask=None
    ):
        super().__init__()

        self.sem_attention = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim, num_node, edge,culter_size, feed_forward_dim, num_heads, dropout, mask,dtw_mask
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




class Pooler(nn.Module):
    '''Pooling the token representations of region time series into the region level.
    '''

    def __init__(self, region_size, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model
        """
        super(Pooler, self).__init__()

        ## attention matirx
        self.att = nn.Linear(d_model, region_size)

        self.d_model = d_model
        self.region_size = region_size
        if agg == 'avg':
            self.agg = nn.AdaptiveAvgPool2d(((region_size, None)))
        elif agg == 'max':
            self.agg = nn.AdaptiveMaxPool2d(((region_size, None)))
        else:
            raise ValueError('Pooler supports [avg, max]')

    def forward(self, x):
        """
        :param x: key sequence of region embeding, nlvc
        :return x_agg: region embedding for spatial similarity, nvc
        """
        # calculate the attention matrix A using key x
        A = self.att(x)

        A = F.softmax(A, dim=2)

        # calculate region embeding using attention matrix A
        x = torch.einsum('nlvq,nlvc->nlqc', A, x)
        x_agg = self.agg(x)  # ncqv->ncv

        return  x_agg
"""
pool=Pooler(20,64)
x=torch.rand(64,12,325,64)
y=pool(x)
print(y.shape)
"""

