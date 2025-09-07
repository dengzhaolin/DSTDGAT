import torch.nn as nn
import torch

DEVICE1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE2 = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

from Module.Embeding import *

#from Module.SemanticGraphTransformer import Sem_Graph_Encoder

from Module.External_Attention import Sem_Graph_Encoder1
#from Module.External_Attention1 import Sem_Graph_Encoder1
#from 新建模型.External_Attention import Sem_Graph_Encoder1
#from 新建模型.Temporal_block import Temporal_Attention,TemporLearning ,Sem_Graph_Encoder1
#from 新建模型.Spatial_Temporal_Aware import Sem_Graph_Encoder1

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

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
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



class SelfAttentionLayer1(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0., mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
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
        b,l,n,c=x1.shape

        x1 = x1.transpose(1, -2)

        x1=x1.reshape(b*n,l,c)


        # x: (batch_size, ..., length, model_dim)
        residual = x1

        out = self.attn(x1,x1,x1)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out=out.reshape(b,n,l,c)

        out = out.transpose(1, -2)
        return out
class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[(day_emb[:, :, :] * self.time).type(torch.LongTensor)]
        time_day = time_day.transpose(1, 2).contiguous()

        week_emb = x[..., 2]
        time_week = self.time_week[(week_emb[:, :, :]).type(torch.LongTensor)]
        time_week = time_week.transpose(1, 2).contiguous()

        tem_emb = time_day + time_week

        tem_emb = tem_emb.permute(0,2,1,3)

        return tem_emb

class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        model_dim=64,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        adj=None,
        edge=None,
        dtw_adj=None
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim=model_dim

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.layer=num_layers

        self.Temb = TemporalEmbedding(288, model_dim)

        self.start_conv = nn.Conv2d(
            in_channels=3, out_channels=model_dim, kernel_size=(1, 1)
        )


        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)


        self.embeding=DataEmbedding(1,self.model_dim,num_nodes)




        self.attn_layers_t = nn.ModuleList(
            [
                #Temporal_Attention(self.model_dim*2,feed_forward_dim,num_heads,dropout)
                #TemporLearning(self.model_dim*2,feed_forward_dim,num_heads,dropout)
                SelfAttentionLayer1(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )








        #self.attn_layers_s = Sem_Graph_Encoder1(self.model_dim*2,num_layers,num_nodes,adj,20,edge,feed_forward_dim,num_heads,dtw_mask=dtw_adj)
        self.attn_layers_s = Sem_Graph_Encoder1(self.model_dim , num_layers, num_nodes, adj, 20, edge,
                                                feed_forward_dim, num_heads, dtw_mask=dtw_adj)








    def forward(self, x,adj):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)


        x=self.embeding(x,adj)




        batch_size = x.shape[0]



        
        for attn in self.attn_layers_t:
            x = attn(x)


        x = self.attn_layers_s(x)

        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out



