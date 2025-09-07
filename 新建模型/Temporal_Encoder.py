
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from 新建模型.graph import *
from .PGAT1 import *

class Temporal(nn.Module):
    def __init__(self,model_dim,feed_forward_dim,head_num,num_nodes,dropout,emb1,emb2,emb3,emb4):
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
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        self.cov=nn.Sequential(*conv)



        self.gcn1 = DGCN2(model_dim, num_nodes,head_num, emb=emb1)
        self.gcn2 = DGCN2(model_dim, num_nodes,head_num, emb=emb2)
        self.gcn3 = DGCN2(model_dim, num_nodes,head_num, emb=emb3)
        self.gcn4 = DGCN2(model_dim, num_nodes, head_num, emb=emb4)


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
        x10_12 = x[..., 9:]

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
        out=self.gcn4(out)




        return out


class Temporal_(nn.Module):
    def __init__(self,model_dim,num_layer,num_nodes,feed_forward_dim=2048, num_heads=4, dropout=0.1,cheb_polynomials=None
    ):
        super().__init__()
        emb1 = nn.Parameter(torch.randn(model_dim, num_nodes, 3))
        emb2 = nn.Parameter(torch.randn(model_dim, num_nodes, 3))
        emb3 = nn.Parameter(torch.randn(model_dim, num_nodes, 6))
        emb4 = nn.Parameter(torch.randn(model_dim, num_nodes, 12))

        self.sem_attention = nn.ModuleList(
            [
                Dilated_Cov(model_dim, num_heads,num_nodes, dropout,emb1,emb3,emb4,cheb_polynomials
                                   )
                for _ in range(num_layer)
            ]
        )
        self.layer=num_layer




    def forward(self,x):

        for i in range(self.layer):

            x =self.sem_attention[i](x)


        return x



class Dilated_Cov(nn.Module):
    def __init__(self,model_dim,head_num,num_nodes,dropout,emb1,emb2,emb3,cheb_polynomials):
        super().__init__()

        self.spitt=Splitting1()

        self.dropout=dropout




        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        Conv5 = []
        Conv6 = []

        pad_l = 3
        pad_r = 3

        k1 = 3
        k2 = 5
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
        conv = []
        conv += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(model_dim, model_dim, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        self.cov = nn.Sequential(*conv)


        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)
        self.conv5 = nn.Sequential(*Conv5)
        self.conv6 = nn.Sequential(*Conv6)

        """
        self.gcn1 = DGCN2(model_dim, num_nodes, head_num, emb=emb1)
        self.gcn2 = DGCN2(model_dim, num_nodes, head_num, emb=emb2)
        self.gcn3 = DGCN2(model_dim, num_nodes, head_num, emb=emb3)
        
        
        
        
        

        self.gcn1 = DGCN3(model_dim, num_nodes, head_num, emb=emb1)
        self.gcn2 = DGCN3(model_dim, num_nodes, head_num, emb=emb1)
        self.gcn3 = DGCN3(model_dim, num_nodes, head_num, emb=emb1)
        self.gcn4 = DGCN3(model_dim, num_nodes, head_num, emb=emb1)
        self.gcn5 = DGCN3(model_dim, num_nodes, head_num, emb=emb2)
        self.gcn6 = DGCN3(model_dim, num_nodes, head_num, emb=emb2)
        self.gcn = DGCN3(model_dim, num_nodes, head_num, emb=emb3)
        """
        self.gcn1 = Diffusion_GAT2(model_dim, head_num, num_nodes, dropout=0.1, emb=emb1,
                                   cheb_polynomials=cheb_polynomials)
        self.gcn2 = Diffusion_GAT2(model_dim, head_num, num_nodes, dropout=0.1, emb=emb1,
                                   cheb_polynomials=cheb_polynomials)
        self.gcn3 = Diffusion_GAT2(model_dim, head_num, num_nodes, dropout=0.1, emb=emb1,
                                   cheb_polynomials=cheb_polynomials)
        self.gcn4 = Diffusion_GAT2(model_dim, head_num, num_nodes, dropout=0.1, emb=emb1,
                                   cheb_polynomials=cheb_polynomials)
        self.gcn5 = Diffusion_GAT2(model_dim, head_num, num_nodes, dropout=0.1, emb=emb2,
                                   cheb_polynomials=cheb_polynomials)
        self.gcn6 = Diffusion_GAT2(model_dim, head_num, num_nodes, dropout=0.1, emb=emb2,
                                   cheb_polynomials=cheb_polynomials)
        self.gcn = Diffusion_GAT2(model_dim, head_num, num_nodes, dropout=0.1, emb=emb3,
                                  cheb_polynomials=cheb_polynomials)







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


        (x1,x2,x3,x4)=self.spitt(x)

        x1_=self.conv1(x1)
        x1=self.gcn1(x1_)  #1

        x2=self.conv2(x2)
        x2 = self.gcn2(x2)  #1

        x3 = self.conv3(x3)
        x3 = self.gcn3(x3)   #1

        x4 = self.conv4(x4)
        x4 = self.gcn4(x4)     #1

        x1_3=self.concat(x1,x3)

        x1_3=self.conv5(x1_3)

        x1_3=self.gcn5(x1_3)    #2

        x2_4=self.concat(x2,x4)

        x2_4=self.conv6(x2_4)

        x2_4=self.gcn6(x2_4)   #2

        x_1_2_3_4=self.concat(x1_3,x2_4)

        x_1_2_3_4=self.cov(x_1_2_3_4)

        x_1_2_3_4=self.gcn(x_1_2_3_4)  #3


        out=x_1_2_3_4+x

        return out








class Splitting1(nn.Module):
    def __init__(self):
        super(Splitting1, self).__init__()

    def odd1(self, x):
        return x[:, :, :, ::4]

    def odd2(self, x):
        return x[:, :, :, 1::4]
    def odd3(self, x):
        return x[:, :, :, 2::4]
    def odd4(self, x):
        return x[:, :, :, 3::4]

    def forward(self, x):
        return (self.odd1(x), self.odd2(x), self.odd3(x), self.odd4(x))

