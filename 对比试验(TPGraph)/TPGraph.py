# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
# from GCN_models import GCN
# from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np
from Graph_Fusion import get_normalize
from torch.nn import init
from args_parameter_pems import *
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        context = ScaledDotProductAttention()(Q, K, V)
        context = context.permute(0, 3, 2, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)
        output = self.fc_out(context)

        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"


        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)

        context = ScaledDotProductAttention()(Q, K, V)
        context = context.permute(0, 2, 3, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)
        output = self.fc_out(context)
        del context
        return output


class GCN(nn.Module):  # GCN
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, adj):
        flow_x = data
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)
        output_1 = self.linear_1(flow_x)
        output_1 = self.act(torch.matmul(adj.float(), output_1))
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(adj.float(), output_2))

        return output_2


class STransformer(nn.Module):
    def __init__(self, L_W, embed_size, heads, adj1, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.D_S = adj1.to(args.DEVICE)
        self.embed_liner = nn.Linear(adj1.shape[0], embed_size)
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, out):
        self.D_S = out
        B, N, T, C = query.shape
        D_S = self.embed_liner(self.D_S)
        D_S = D_S.expand(B, T, N, C)
        D_S = D_S.permute(0, 2, 1, 3)
        query = query + D_S
        attention = self.attention(query, query, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))
        del D_S
        return U_S


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        self.time_num = time_num
        self.TE_fuse = nn.Conv2d(2 * (args.num_his + args.num_week + args.num_day),
                                 (args.num_his + args.num_week + args.num_day), 1)

        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, TH):
        query = torch.cat((query, TH), dim=-2)
        attention = self.attention(query, query, query)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.TE_fuse((forward + x).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = self.dropout(self.norm2(out))
        return out


### STBlock

class STTransformerBlock(nn.Module):
    def __init__(self, L_W, embed_size, heads, adj1, adj2, time_num, cheb_K, dropout, forward_expansion, device):
        super(STTransformerBlock, self).__init__()

        self.STransformer = STransformer(L_W, embed_size, heads, adj1, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, out, TH):
        x1 = self.norm1(self.STransformer(value, key, query, out) + query)
        x2 = self.dropout(self.norm2(self.TTransformer(x1, x1, x1, TH) + x1))
        return x2


class Encoder(nn.Module):
    def __init__(
            self,
            L_W,
            embed_size,
            num_layers,
            heads,
            adj1,
            adj2,
            time_num,
            device,
            forward_expansion,
            cheb_K,
            dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.adj1 = adj1
        self.adj2 = adj2
        self.device = device
        self.weight = nn.Parameter(torch.randn(2))
        self.sigmoid = nn.Sigmoid()
        self.gcn = GCN(in_c=embed_size, hid_c=embed_size * 4, out_c=embed_size)
        self.norm_adj = nn.InstanceNorm2d(1)
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    L_W,
                    embed_size,
                    heads,
                    adj1,
                    adj2,
                    time_num,
                    cheb_K,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    device=device
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layer = num_layers
        self.dropout = nn.Dropout(dropout)
        self.upE = nn.Conv2d(args.num_pred, args.num_his + args.num_week + args.num_day, 1)
        self.upP = nn.Conv2d(args.num_pred, args.num_his + args.num_week + args.num_day, 1)

    def forward(self, x, TE):
        ''''Multi-Graph convolution'''
        A = self.adj1.to(self.device)  # Graph 1
        B = self.adj2.to(self.device)  # Graph 2
        A1 = get_normalize(A)
        B1 = get_normalize(B)
        w1, w2 = self.sigmoid(self.weight)
        out_adj = w1 * A1 + w2 * B1
        B, N, T, C = x.shape
        X_G = torch.Tensor(B, N, 0, C).to(args.DEVICE)
        for k in range(x.shape[2]):
            o = self.gcn(x[:, :, k, :], out_adj)
            o = o.unsqueeze(2)
            X_G = torch.cat((X_G, o), dim=2)
        X_G = self.dropout(X_G + x)
        TE_his = TE[:, :args.num_his]
        TE_pre = TE[:, args.num_his:]
        TH = self.upE(TE_his).permute(0, 2, 1, 3)
        TP = self.upP(TE_pre).permute(0, 2, 1, 3)

        # Last layer would encoder the predition temporal features
        for ind, layer in enumerate(self.layers, 1):
            X_G = layer(X_G, X_G, X_G, out_adj, TP) if ind == self.num_layer else layer(X_G, X_G, X_G, out_adj, TH)
        return X_G


class Transformer(nn.Module):
    def __init__(
            self,
            adj1,
            adj2,
            L_W,
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            cheb_K,
            dropout,

            device=args.DEVICE
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            L_W,
            embed_size,
            num_layers,
            heads,
            adj1,
            adj2,
            time_num,
            device,
            forward_expansion,
            cheb_K,
            dropout
        )
        self.device = device

    def forward(self, src, TE):
        enc_src = self.encoder(src, TE)
        return enc_src


class selfAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=0):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(selfAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, args.num_his + args.num_week + args.num_day)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)
        return out


class FCN(nn.Module):
    def __init__(self, T_dim):
        super(FCN, self).__init__()

        self.at_c = selfAttention(d_model=T_dim[0] + T_dim[1], d_k=T_dim[0] + T_dim[1], d_v=T_dim[0] + T_dim[1], h=8)
        self.at_d = selfAttention(d_model=T_dim[0] + T_dim[2], d_k=T_dim[0] + T_dim[2], d_v=T_dim[0] + T_dim[2], h=8)

        self.w_c = nn.Parameter(torch.randn(1))
        self.w_d = nn.Parameter(torch.randn(1))

    def forward(self, xc, xd, xw):
        xc = xc.squeeze(3).permute(0, 2, 1)
        xd = xd.squeeze(3).permute(0, 2, 1)
        xw = xw.squeeze(3).permute(0, 2, 1)

        xcd = torch.cat([xc, xd], 2)
        xcw = torch.cat([xc, xw], 2)
        xcdw = torch.cat([xc, xd, xw], 2)

        xc = self.at_c(xcd, xcd, xcd)
        xd = self.at_d(xcw, xcw, xcw)
        xc = xc.permute(0, 2, 1).unsqueeze(3)
        xd = xd.permute(0, 2, 1).unsqueeze(3)
        xw = xcdw.permute(0, 2, 1).unsqueeze(3)

        out = torch.add(self.w_c * xc, self.w_d * xd)
        out = out + xw

        return out

# Temporal features embedding
class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x.to('cuda:0'))
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    '''
    :param  TE: [b_s, num_his + num_pred, 2] (dayofweek, timeofday)
    :param  T: num of time steps in one day
    :param  D: output dims
    :return  : [b_s, num_his + num_pred, 1, D]
    '''

    def __init__(self, D, bn_decay):
        super(STEmbedding, self).__init__()

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

    def forward(self, TE, T=288):
        # time embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE)
        del dayofweek, timeofday
        return TE


class TPGraph(nn.Module):
    '''
    :param in_channels = 1 # Channels of input
    :param embed_size = 64 # Dimension of hidden embedding features
    :param time_num = 288
    :param num_layers = 2 # Number of ST Block
    :param T_dim = 12 # Input length
    :param output_T_dim = 12 # Expected prediction length
    :param heads = 2 # Number of Heads in MultiHeadAttention
    :param cheb_K = 2
    :param forward_expansion = 4 # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    :param dropout = 0  # default = 0
    '''

    def __init__(
            self,
            adj1,
            adj2,
            L_W,
            in_channels,
            embed_size,
            time_num,
            num_layers,
            T_dim,
            output_T_dim,
            heads,
            cheb_K,
            forward_expansion,
            device,
            dropout
    ):
        super(TPGraph, self).__init__()
        self.T_dim = T_dim
        self.device = device
        self.forward_expansion = forward_expansion
        self.FCN = FCN(self.T_dim)
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.bn1 = nn.BatchNorm2d(embed_size, momentum=0.1)
        self.Transformer = Transformer(
            adj1,
            adj2,
            L_W,
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            cheb_K,
            dropout=dropout
        )

        self.conv2 = nn.Conv2d(np.array(T_dim).sum(), output_T_dim, 1)
        self.bn2 = nn.BatchNorm2d(output_T_dim, momentum=0.1)
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.up = nn.Conv2d(args.num_his, 16, 1)
        self.TEmbedding = STEmbedding(embed_size, bn_decay=0.1)
        self.fc1 = nn.Linear(args.num_his, np.array(T_dim).sum())
        self.fc2 = nn.Linear(args.num_pred, np.array(T_dim).sum())

    def forward(self, xc, xd, xw, te):
        B, C, N, T = xc.shape
        TE = self.TEmbedding(te).expand(B, 24, N, -1)
        xc = xc.permute(0, 3, 2, 1)
        xd = xd.permute(0, 3, 2, 1).to(self.device)
        xw = xw.permute(0, 3, 2, 1).to(self.device)

        # Mutlti-scale temporal features fusion
        input_Transformer = self.FCN(xc, xd, xw)

        input_Transformer = input_Transformer.permute(0, 3, 2, 1)
        input_Transformer = F.relu(self.bn1(self.conv1(input_Transformer)))

        input_Transformer = input_Transformer.permute(0, 2, 3, 1)

        # Multi-graph convolution & Dynamic spatial-temporal prediction
        output_Transformer = self.Transformer(input_Transformer, TE)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        out = F.relu(self.bn2(self.conv2(output_Transformer)))
        out = out.permute(0, 3, 1, 2)
        out = self.conv3(out)
        out = out.squeeze(1)

        return out  # [B, output_dim, N]
adj=torch.randn(325,325)
extra_feature = torch.Tensor(120, 2)
tp=TPGraph(adj,adj,extra_feature,1,64,288,1,12,12,4,2,4,torch.device(device='cpu'),0.1)