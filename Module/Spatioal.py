import torch

from Module.attention import *
from Module.graph import *
from Module.mokuai import *
from Module.Afir import *

class SpatioLearning(nn.Module):
    def __init__(self,input_dim,feed_forward_dim,layer,num_head,len,cluter_size,dropout):
        super().__init__()

        self.layer=layer
        self.post_embeding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(len,input_dim))
            )
        self.SpaAtt = nn.ModuleList(
            [
                SelfAdaAttentionLayer_cluter(input_dim,len,cluter_size,feed_forward_dim,num_head,dropout)
                for _ in range(layer)
            ]
        )

    def forward(self,x):

        b,l,n,c=x.shape

        x=self.post_embeding.expand(b,l,*self.post_embeding.shape) + x

        for layer in self.SpaAtt:

            x=layer(x,dim=2)

        return x


"""
TEM=SpatioLearning(64,256,2,8,325,20,0.2)
x=torch.rand(64,12,325,64)

y=TEM(x)
print(y)

"""


