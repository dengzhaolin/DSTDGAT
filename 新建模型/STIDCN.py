import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import nn, sum
from torch.nn import init
from Module.Embeding import DataEmbedding
import math
#from 新建模型.Temporal_Encoder_xiaorong import *
from 新建模型.Temporal_Encoder import *
from 新建模型.graph import *
from 新建模型.multi import *
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))
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
class GEANet1(nn.Module):

    def __init__(
            self, dim, n_heads,meta_axis,embed_dim,output_dim):
        super().__init__()

        self.dim = dim
        self.external_num_heads =n_heads
        self.meta_axis =meta_axis
        self.unit_size = self.dim//n_heads

        assert self.unit_size * self.external_num_heads == self.dim, "dim must be divisible by external_num_heads"

        # self.q_Linear = nn.Linear(in_dim, gconv_dim - dim_pe)
        self.node_U1 = nn.Linear(self.unit_size, self.unit_size,bias=False)
        self.node_U2 = nn.Linear(self.unit_size, self.unit_size,bias=False)




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

class IDGCN(nn.Module):
    def __init__(
            self,
            channels=64,
            diffusion_step=1,
            splitting=True,
            num_nodes=170,
            dropout=0.2, emb=None
    ):
        super(IDGCN, self).__init__()


        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3
        pad_r = 3

        k1 = 5
        k2 = 3
        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)




        #self.dgcn = GEANet(channels,4,True,channels,channels)
        #self.dgcn=DGCN(channels, num_nodes, diffusion_step, dropout, emb)

        self.dgcn=DGCN(channels, num_nodes, diffusion_step, dropout, emb)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x


        """



        x1 = self.conv1(x_even).permute(0,3,2,1)
        x1 = self.dgcn(x1,spatial_embedding).permute(0,3,2,1)
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.conv2(x_odd).permute(0,3,2,1)
        x2 = self.dgcn(x2,spatial_embedding).permute(0,3,2,1)
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c).permute(0,3,2,1)
        x3 = self.dgcn(x3,spatial_embedding).permute(0,3,2,1)
        x_odd_update = d + x3

        x4 = self.conv4(d).permute(0,3,2,1)
        x4 = self.dgcn(x4,spatial_embedding).permute(0,3,2,1)
        x_even_update = c + x4

        return (x_even_update, x_odd_update)
        
        """
        x1 = self.conv1(x_even)
        x1 = self.dgcn(x1)
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.conv2(x_odd)
        x2 = self.dgcn(x2)
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c)
        x3 = self.dgcn(x3)
        x_odd_update = d + x3

        x4 = self.conv4(d)
        x4 = self.dgcn(x4)
        x_even_update = c + x4

        return (x_even_update, x_odd_update)


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


class DGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator =Graph_Generator(channels,num_nodes,diffusion_step,dropout)# GEANet(channels,4)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x):
        skip = x
        x = self.conv(x)

        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x * self.emb + skip#
        return x
class IDGCN_Tree(nn.Module):
    def __init__(
            self, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1
    ):
        super().__init__()

        self.memory1 = nn.Parameter(torch.randn(channels, num_nodes, 6))

        self.memory2 = nn.Parameter(torch.randn(channels, num_nodes, 3))
        self.memory3 = nn.Parameter(torch.randn(channels, num_nodes, 3))

        self.IDGCN1 = IDGCN(
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout, emb=self.memory1
        )
        self.IDGCN2 = IDGCN(

            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout, emb=self.memory2
        )
        self.IDGCN3 = IDGCN(

            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout, emb=self.memory2
        )

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x):
        x_even_update1, x_odd_update1 = self.IDGCN1(x)
        x_even_update2, x_odd_update2 = self.IDGCN2(x_even_update1)
        x_even_update3, x_odd_update3 = self.IDGCN3(x_odd_update1)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        output = concat0 + x
        return output
class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
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

        tem_emb = tem_emb.permute(0,3,1,2)

        return tem_emb
class STIDGCN(nn.Module):
    def __init__(
            self,  input_dim, num_nodes, channels, dropout=0.1,cheb_polynomials=None
    ):
        super().__init__()


        self.num_nodes = num_nodes
        self.output_len = 12
        diffusion_step = 4

        self.Temb = TemporalEmbedding(288,channels)

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )

        self.tree = IDGCN_Tree(
            channels=channels*2 ,
            diffusion_step=diffusion_step,
            num_nodes=self.num_nodes,
            dropout=dropout,
        )

        self.glu = GLU(channels*2, dropout)

        self.regression_layer = nn.Conv2d(
            channels*2 , self.output_len, kernel_size=(1, self.output_len)
        )

        self.embedding=nn.Linear(num_nodes,2*channels)

        self.encoder=Temporal_(channels*2,1,num_nodes,128,4,dropout,cheb_polynomials)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input):

        # Encoder
        # Data Embedding
        x=input.permute(0, 3, 2, 1)

        time_emb = self.Temb(input)
        x = torch.cat([self.start_conv(x)] + [time_emb], dim=1)
        # IDGCN_Tree
        #spatial_embedding=self.embedding(adj)
        #x = self.tree(x)

        x=self.encoder(x)

        # Decoder
        gcn = self.glu(x) + x
        prediction = self.regression_layer(F.relu(gcn))
        return prediction


"""
x=torch.randn(64,12,325,64)

s=STIDGCN(3,325,64)

y=s(x)
print(y.shape)
"""