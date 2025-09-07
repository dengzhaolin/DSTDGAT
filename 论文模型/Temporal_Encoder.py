

import torch
import torch.nn as nn
from 新建模型.graph import *

class Temporal(nn.Module):
    def __init__(self,model_dim,feed_forward_dim,head_num,num_nodes,dropout,emb1,emb2,emb3):
        super().__init__()

        self.dropout=dropout

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        Conv5 = []
        Conv6 = []
        pad_l = 3
        pad_r = 3

        k1 = 5
        k2 = 3
        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]


        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv5 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        Conv6 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)
        self.conv5 = nn.Sequential(*Conv5)
        self.conv6 = nn.Sequential(*Conv6)
        conv = []
        conv+= [
            nn.Conv2d(model_dim, model_dim, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, 1)),
            nn.Tanh(),
        ]

        self.cov=nn.Sequential(*conv)



        self.gcn1 = DGCN2(model_dim, num_nodes,head_num, emb=emb1)
        self.gcn2 = DGCN2(model_dim, num_nodes,head_num, emb=emb2)
        self.gcn3 = DGCN2(model_dim, num_nodes,head_num, emb=emb3)


        self.sigmoid=nn.Sigmoid()

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()
    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)



    def forward(self,x):


        x1_3=x[...,0:3]
        x4_6 = x[..., 3:6]
        x7_9 = x[..., 6:9]
        x10_12 = x[..., 9:-1]

        x1_3=self.conv1(x1_3)
        x1_3=self.gcn1(x1_3)
        x4_6 = self.conv2(x4_6)
        x4_6=self.gcn1(x4_6)
        x7_9 = self.conv3(x7_9)
        x7_9=self.gcn2(x7_9)
        x10_12 = self.conv4(x10_12)
        x10_12=self.gcn2(x10_12)

        x1_6=torch.cat((x1_3,x4_6),-1)
        x7_12 = torch.cat((x7_9, x10_12), -1)

        x1_6=self.conv5(x1_6)

        x1_6=self.gcn3(x1_6)

        x7_12=self.conv6(x7_12)

        x7_12=self.gcn3(x7_12)

        x1_12=torch.cat((x1_6, x7_12), -1)

        out=self.cov(x1_12)


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

class Temporal_(nn.Module):
    def __init__(self,model_dim,num_layer,num_nodes,feed_forward_dim=2048, num_heads=8, dropout=0
    ):
        super().__init__()
        emb1 = nn.Parameter(torch.randn(model_dim, num_nodes, 3))
        emb2 = nn.Parameter(torch.randn(model_dim, num_nodes, 3))
        emb3 = nn.Parameter(torch.randn(model_dim, num_nodes, 6))

        self.sem_attention = nn.ModuleList(
            [
                Temporal(model_dim, feed_forward_dim, num_heads,num_nodes, dropout,emb1,emb2,emb3
                                   )
                for _ in range(num_layer)
            ]
        )
        self.layer=num_layer




    def forward(self,x):

        for i in range(self.layer):

            x =self.sem_attention[i](x)


        return x







