import torch

from Module.Temporal import *
from Module.Spatioal import *
from Module.Embeding import *
class AGFomer(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_head,num_layers,seq_len,adj,dropout):
        super().__init__()

        self.use_mixed_proj=True

        self.feed_forward_dim = 256

        self.hidden_dim=hidden_dim

        self.output_dim=output_dim

        self.linear=nn.Linear(input_dim,hidden_dim)
        self.tod_embedding_dim = 32
        self.dow_embedding_dim = 32
        self.spatial_embedding_dim = 32
        self.adaptive_embedding_dim=80

        self.model_dim = 96

        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                seq_len * self.model_dim, seq_len * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(seq_len, seq_len)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(288, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(325, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(12, 325, 80)))


        self.layers_t = nn.ModuleList(
            [
                TemporLearning1(self.model_dim,self.feed_forward_dim,num_head,seq_len,adj,dropout)
                for _ in range(num_layers)
            ]
        )

        self.layers_s = nn.ModuleList(
            [
                SpatioLearning1(self.model_dim,self.feed_forward_dim,num_head,seq_len,adj,dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm_t=nn.LayerNorm(hidden_dim)
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(self.model_dim, hidden_dim)
        self.input_proj=nn.Linear(1,32)

        self.embeding=DataEmbedding(1,self.model_dim,64)
    def forward(self,x):
        b,l,n,c=x.shape
        Tem_nodevec1 = nn.Parameter(torch.randn(l, 10), requires_grad=True)
        Tem_nodevec2 = nn.Parameter(torch.randn(10, l), requires_grad=True)
        Spa_nodevec1 = nn.Parameter(torch.randn(n, 10), requires_grad=True)
        Spa_nodevec2 = nn.Parameter(torch.randn(10, n), requires_grad=True)

        features=[self.input_proj(x[...,:1])]

        support_t = [Tem_nodevec1, Tem_nodevec2]
        support_s = [Spa_nodevec1, Spa_nodevec2]
        """
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (x[...,1] * 288).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                x[...,2].long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(b, *self.adaptive_embedding.shape)
            )
            #features.append(adp_emb)


        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        """

        x=self.embeding(x)



        for Temlayer in self.layers_t:
            x = Temlayer(x,support_t)


        for Spalayer in self.layers_s:
            x = Spalayer(x,support_s)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                b, n, l * self.model_dim
            )
            out = self.output_proj(out).view(
                b, n, l, self.output_dim
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
"""
adj=torch.rand(325,325)
Ag=AGFomer(3,64,1,8,2,12,adj,0.2)
x=torch.rand(64,12,325,3)
y=Ag(x)
print(y)
"""

