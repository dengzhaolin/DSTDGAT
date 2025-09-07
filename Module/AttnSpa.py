import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Module.AttTem import *

class AdaptiveGCN(nn.Module):
    def __init__(self, channels, order=2, include_self=True,  is_adp=True):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1
        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x,support_adp):
        B,L,N,C=x.shape

        x=x.permute(0,2,1,3)



        if N == 1:
            return x
        if self.is_adp:
            nodevec1 = support_adp[0]
            nodevec2 = support_adp[1]

        else:
            support = support_adp
        x = x.permute(0, 1, 3, 2).reshape(B * L, C, N)
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        if self.is_adp:
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=-1)
            support = support + [adp]
        for a in support:
            a=a.to(x.device)
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        out = out.reshape(B, L, N, C)
        return out
def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class Attn_spa(nn.Module):
    def __init__(self, dim, seq_len, k, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, x_time=None):
        d_h, h, k = self.dim_head, self.heads, self.k

        B,l,n,c=x.shape

        x=x.reshape(B*l,n,c)

        b, n, c =x.shape

        if x_time is not None:
            x_time=x_time.reshape(b,n,c)

        v_len = n if x_time is None else x_time.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if x_time is None else x_time
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if x_time is None else x_time
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(B,l, n, -1)
        out= self.to_out(out)

        return out


class SpatialLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, is_adp, k, is_cross):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = SpaDependLearning(channels, nheads=nheads, order=order, seq_len=target_dim,
                                               include_self=include_self,  is_adp=is_adp,
                                               k=k, is_cross=is_cross)

    def forward(self, y,  support, itp_y=None):
        B, L, N, C = y.shape
        if N == 1:
            return y
        y = self.feature_layer(y,  support, itp_y)
        return y


class SpaDependLearning(nn.Module):
    def __init__(self, channels, nheads, seq_len, order, include_self,  is_adp,  k,
                 is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self,  is_adp=is_adp)
        self.attn = Attn_spa(dim=channels, seq_len=seq_len, k=k, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.LayerNorm( channels)
        self.norm1_attn = nn.LayerNorm( channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.LayerNorm( channels)

    def forward(self, y, support, itp_y=None):
        B,L,N,C=y.shape
        y_in1 = y

        y_local = self.GCN(y, support)  # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)
        y_attn = y
        if self.is_cross:
            itp_y_attn = itp_y
            y_attn = self.attn(y_attn, itp_y_attn)
        else:
            y_attn = self.attn(y_attn)

        y_attn = y_in1 + y_attn
        y_attn = self.norm1_attn(y_attn)

        y_in2 = y_local + y_attn
        y = F.relu(self.ff_linear1(y_in2))
        y = self.ff_linear2(y)
        y = y + y_in2

        y = self.norm2(y)
        return y

"""
x=torch.rand(64,12,170,64)
at=SpatialLearning(64,8,170,170,2,'True',170,'True',)
y=at(x)
print(y.shape) ###64,12,170,64
"""
class FeatureFousion(nn.Module):
    def __init__(self,channels,chek,num_node,nheads):
        super(FeatureFousion,self).__init__()
        #self.MPNN = AdaptiveGCN(channels, chek)
        self.AttenSpa = SpatialLearning(channels,nheads,num_node,chek,'True','True',nheads,False)
        self.AttenTem = TemporalLearning(channels, nheads, is_cross=False)  ### is_cross是否使用x_time

        self.norm1=nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)


    def forward(self, x, support_adp, x_time=None):
        B, L,N,C= x.shape

        y = x

        y = self.AttenTem(y,x_time)
        y = self.AttenSpa(y,  support_adp, x_time)  # (B,channel,K*L)
        y=y.reshape(B,-1,N*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)


        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(B,L,N,-1)
        residual = residual.reshape(B,L,N,-1)
        skip = skip.reshape(B,L,N,-1)

        return (x + residual) / math.sqrt(2.0), skip


