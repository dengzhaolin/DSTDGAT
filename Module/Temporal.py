import torch

from Module.attention import *

from Module.mokuai import *
#from Module.Afir import *
#from Module.FixationTemporalCorrelation import *
class TemporLearning(nn.Module):
    def __init__(self,input_dim,feed_forward_dim,num_head,seq_len,dropout):
        super().__init__()

        self.Global_Relational = SelfAttentionLayer(input_dim, feed_forward_dim, num_head, dropout, mask=True)
        self.TemGcn = GCN(input_dim, input_dim, seq_len, adj=None, mode='temporal')

        self.Ada_Mix = AdaptiveMixtureUnits(input_dim, dropout)

        self.Global_norm = nn.LayerNorm(input_dim)

        self.local_norm = nn.LayerNorm(input_dim)

        self.pos_embedding= nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(seq_len, input_dim))
            )

        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(input_dim * 2, input_dim)

        self.revin=RevIN(input_dim)




    def forward(self,x,adj):


        global_out = self.Global_Relational(x,1)

        global_out=self.Global_norm(global_out)

        local_out = self.TemGcn(x,adj)
        """
        b, l, n, c = x.shape
        x_ = x.permute(0, 2, 1, 3).reshape(b * n, l, c)

        x_, (_, _) = self.lstm(x_)

        #

        x_ = self.linear(x_)

        x_=x_.reshape(b,l,n,c)
        """

        local_out=self.local_norm(local_out)


        out=self.Ada_Mix(x,global_out,local_out)



        return out




"""
TEM=TemporLearning(64,256,8,12,0.2)
x=torch.rand(64,12,325,64)


y=TEM(x)
print(y)

"""






