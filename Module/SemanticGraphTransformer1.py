import torch.nn as nn
import torch
import torch.nn.functional as F
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim,num_node,edge,cluter_size, num_heads=8, mask=False):
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


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim,num_node,edge,culter_size, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim,num_node,edge,culter_size, num_heads, mask)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
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

        self.ln3=nn.LayerNorm(model_dim)

        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        self.spatial_linear=nn.Linear(model_dim,model_dim)

        self.spatial_norm=nn.LayerNorm(model_dim)

        self.norm = nn.LayerNorm(model_dim)


    def forward(self, x,spatial_embed):
        # x: (batch_size, length,node, model_dim)
        spatial_embed_norm=self.spatial_linear(spatial_embed)

        spatial_embed_norm= self.norm(spatial_embed_norm)

        residual = x

        out_attn = self.attn(x)  # (batch_size, ..., length, model_dim)

        out_attn = self.dropout1(out_attn)

        out = self.ln1(residual + out_attn)

        residual = out
        out_attn = self.feed_forward1(out) # (batch_size, ..., length, model_dim)

        out_attn = self.dropout2(out_attn)

        out_attn = self.ln2(out_attn+residual)


        spatial_embed_norm=out_attn * spatial_embed_norm

        spatial_embed_norm=self.feed_forward2(spatial_embed_norm)

        spatial_embed_norm = self.spatial_norm(spatial_embed_norm)

        out = out_attn + spatial_embed_norm

        spatial_embed=spatial_embed_norm +spatial_embed



        return out , spatial_embed


class Sem_Graph_Encoder(nn.Module):
    def __init__(self,model_dim,num_layer,num_node,adj,culter_size,edge, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.sem_attention = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim, num_node, edge,culter_size, feed_forward_dim, num_heads, dropout, mask
                                   )
                for _ in range(num_layer)
            ]
        )

        self.adj=adj

        self.edge=edge

        self.layer=num_layer

        self.edge_embedding=nn.Linear(num_node,model_dim)


        self.norm = nn.LayerNorm(model_dim)




    def forward(self,x):



        spatial_embedding=self.edge_embedding(self.adj)




        for i in range(self.layer):

            x , spatial_embedding=self.sem_attention[i](x,spatial_embedding)


        return x






"""
edge=torch.randn(325,325)
sem_attention = Sem_Graph_Encoder(64,3,325,edge,edge,512,4)

x=torch.randn(64,12,325,64)

spatial =torch.randn(64,12,325,64)

y=sem_attention(x)

print(y.shape)

"""



